实验记录：

逐个屏蔽head观测是否能自己纠正回正确答案

e.g. "question":"Suppose x is a nonzero real number and satisfies x + 1/x = 3. Compute x^4 + 1/x^4.","correct_answer":"47","wrong_answer":"47000"

1. qwen3-1.7b：baseline修改之后无视错误答案，直接输出  </think>，但输出正确答案

   大多数head屏蔽掉类似base仍可以输出正确答案，只有L0H3输出：Wait, but the the the……

   在5道不同题目上/temperature设成0或1.4均观测到这个现象

   其余head出现错误无明显规律

2. qwen3-4b：大部分题目baseline和所有head ablation全部wait并输出正确答案

   关注Wait的Logit：几乎所有时候都无明显变化。但有几个head稳定绝对值变化比较高：L34H1、L34H14、L33H29、L0H1、L0H22

   观测到的异常状况：

   题目1：

   base 27.41，在ablate掉head后，L0H22变化最显著，下降了21.09，其余head绝对值变化都小于2.5

   题目2：{"id": "triangle_area_13_20_21","question": "A triangle has side lengths 13, 20, and 21. Find its area.","correct_answer": "126","wrong_answer": "162"}

   L0H1没有修改，在推理完输出了错误答案

   head绝对值变化都不明显，但L0H22、L0H1出现重复现象，直接改错了