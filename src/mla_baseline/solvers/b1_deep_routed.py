"""B1-deep-routed — deep-поиск (страницы + реранкер) с роутингом по предмету.

Комбинация двух улучшений: математика идёт чистым B0-путём (поиск ей
систематически вредит во всех реализациях), остальные предметы получают
полный deep-стек. MRO: роутинг из B1Routed, инструмент из B1Deep.
"""

from .b1_deep import B1Deep
from .b1_routed import B1Routed


class B1DeepRouted(B1Routed, B1Deep):
    condition = "b1_deep_routed"
