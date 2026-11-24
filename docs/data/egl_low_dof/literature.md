# 相关工作如何处理这些问题

- Huang & Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (ICCV 2017, arXiv:1703.06868)：在特征空间匹配均值和方差以转移风格。这里匹配的是两路原始策略 RGB 的通道矩，并只混合一半；没有移植其网络，也不能由它推得语义或抓取精度保持。
- Zhao et al., *On Pitfalls of Test-Time Adaptation* (ICML 2023, arXiv:2306.03536)：强调方法选择、评测设置和冻结基线影响适配结论。本轮用固定 alpha、门槛和预算，报告同实例恢复/损害，不在同一批 Dev 上选择更有利的超参数。
- *Adapting Vision-Language-Action Models to Unknown Visual Disruptions During Execution* (SALT, arXiv:2610.07946)：使用前一动作块未执行部分与当前重规划间的重叠做监督。这里 H=E=12，没有那个重叠区间；采用内部噪声验收并做闭环检验，不能声称复现或胜过 SALT。

这些来源提供比较方法和动机；均未证明低维变换安全或本实验会取得正收益。
