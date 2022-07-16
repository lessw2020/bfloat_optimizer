
# pure Bfloat16 optimizer - basic idea is we use Kahan summarization to offset the Bfloat16 precision reduction, allowing full training in BFloat16.

# paper credit - "Revisiting Bfloat16 training" - https://arxiv.org/abs/2010.06192
# original impl - https://github.com/arogozhnikov/adamw_bfloat16




