# 三轮审稿式迭代记录

完整初始接线后，Round0 基础31项CPU通过。以下是实际审查与修复，不把连续跑同一测试称作新迭代。

## Round1：机制与训练数据

发现：专家替代式监督偏离历史实验主教师；仅age0 bootstrap无法训练受保护fresh行；丢弃全部越界目标会饿死局部反馈；按scene/seed分层使半量实际保留7467集。
修改：保留当前448/6/7分频主干，改为同命令配对噪声的fresh Full主教师、方向一致专家辅助；未裁剪endpoint Huber给有界输出方向梯度；开放原始所有年龄窗口；task/style整集半量。
验证：33项CPU通过，包含越界目标仍有正确方向梯度、半量整集统计、上下文/计划P连接、夹爪/fresh保留和历史V9/Graph回归。真实半量样本检查另存SSD诊断回执。GPU未运行。

## Round2

发现并修复：交互向量缺少可重验的真实时间与命令身份；部署previous proposal保存了边界处理之前的版本；回放可能悄悄补零而接受旧schema；专家教师噪声在不同episode中完全相同；训练入口需要复核真实query exposure和已选首waypoint。

新增FP64 interval/int64 command id/连续suffix校验；actual qpos/qvel按配置joint顺序读取；previous proposal绑定实际边界处理后的目标；V10回放schema拒绝旧缺失数据；episode绑定教师底噪；source/current clamping与exposure检查。8条命令历史只包含已经完成的TOPP动作。

验证：71项CPU通过，覆盖真实关节读取、未来/错误命令历史拒绝、native history、Full噪声对齐、优化器/checkpoint/分布式配置回归。首次宽回归暴露旧V7测试把当前架构误当作deployment-only补丁的假设；测试改用实际冻结V7源码构造正例，同时验证当前V10和任何训练模型修改仍被V7兼容门拒绝，生产门禁没有放宽。原失败日志保留。GPU仍未运行。

## Round3

待审查：公平对照、启动与冻结证据、失败/占用门禁、完整回归；GPU验收等待资源。
