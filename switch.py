# =========================
# OSKen 核心模块
# =========================
from os_ken.base import app_manager

# 事件系统（交换机连接、PacketIn等）
from os_ken.controller import ofp_event

# 状态 + 装饰器
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls

# OpenFlow 1.3 协议
from os_ken.ofproto import ofproto_v1_3


# =========================
# 控制器类（必须继承 RyuApp）
# =========================
class SimplePathController(app_manager.RyuApp):

    # 指定 OpenFlow 版本
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    # =========================
    # 工具函数：添加 flow 规则
    # =========================
    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        """
        datapath: 交换机
        priority: 优先级
        match: 匹配条件
        actions: 动作（输出到哪个端口）
        """

        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # 定义动作（执行输出）
        inst = [
            parser.OFPInstructionActions(
                ofproto.OFPIT_APPLY_ACTIONS,
                actions
            )
        ]

        # 构造 flow rule
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst
        )

        # 发送给交换机
        datapath.send_msg(mod)

    # =========================
    # 事件1：交换机连接时触发
    # =========================
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        # 默认规则（table-miss）
        # 👉 所有未知包都发给控制器
        match = parser.OFPMatch()

        actions = [
            parser.OFPActionOutput(
                ofproto.OFPP_CONTROLLER,
                ofproto.OFPCML_NO_BUFFER
            )
        ]

        self.add_flow(datapath, 0, match, actions)

        print(f"✅ 交换机上线 dpid={datapath.id}")

    # =========================
    # 核心函数：决定走哪条路径
    # =========================
    def get_out_port(self, dpid, in_port):
        """
        根据交换机ID + 入端口，决定出端口
        """

        # ========= s1 =========
        if dpid == 1:
            # h1 -> s1
            if in_port == 1:
                return 2   # 👉 发往 s2（路径1）

            # s2 -> s1
            elif in_port == 2:
                return 1   # 👉 回到 h1

        # ========= s2 =========
        elif dpid == 2:
            # s1 -> s2
            if in_port == 1:
                return 2   # 👉 发往 h2

            # h2 -> s2
            elif in_port == 2:
                return 1   # 👉 回到 s1

        # ========= s3 =========
        elif dpid == 3:
            # 当前不使用 s3
            return None

        return None

    # =========================
    # 事件2：交换机不会转发时触发
    # =========================
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):

        msg = ev.msg
        datapath = msg.datapath
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        # 当前交换机 ID（s1=1, s2=2, s3=3）
        dpid = datapath.id

        # 包从哪个端口进来
        in_port = msg.match['in_port']

        # 👉 决策：该往哪个端口发
        out_port = self.get_out_port(dpid, in_port)

        if out_port is None:
            print(f"丢弃 dpid={dpid}, in_port={in_port}")
            return

        # 定义动作
        actions = [parser.OFPActionOutput(out_port)]

        # 安装 flow（以后不用再问控制器）
        match = parser.OFPMatch(in_port=in_port)
        self.add_flow(datapath, 10, match, actions)

        # 发送当前包
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data
        )
        datapath.send_msg(out)

        print(f"➡️ dpid={dpid}: {in_port} → {out_port}")