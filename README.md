# Детекция разладок в МЭГ/ЭЭГ

Исследование данных и сравнение методов сегментации временных рядов.
Основное задание проекта — в `task.pdf`.

| Папка | Назначение |
|---|---|
| `src/` | Алгоритмы: CPOP, регрессионный PELT, RBF-PELT, L1 trend filtering, прямоугольники с GA/PELT и полиномиальная аппроксимация. |
| `utils/` | Загрузка данных (`data_loading.py`), пути проекта (`paths.py`), статистика, визуализации и функции проведения экспериментов. |
| `experiments/` | Ноутбуки и сохранённые результаты. Файлы каждого эксперимента находятся в папке с именем ноутбука без `.ipynb`. |
| `tests/` | Проверки алгоритмов, статистик и визуализаций. |
| `data/` | Исходные Excel-файлы в `excel-data/`, локальный набор `MNE-sample-data/` и первоначальный архив в `archives/`. |

Ноутбуки:

- [eda.ipynb](experiments/eda.ipynb) — исследование обоих источников; таблицы и PDF-атлас в `experiments/eda/`.
- [breakpoint_comparison.ipynb](experiments/breakpoint_comparison.ipynb) — сравнение четырёх подходов и прямоугольного baseline; метрики и графики в `experiments/breakpoint_comparison/`.
- [rectangle_demo.ipynb](experiments/rectangle_demo.ipynb) — прямоугольники, генетический алгоритм и точный PELT; графики встроены в ноутбук.
- [demo.ipynb](experiments/demo.ipynb) — исходный пример с полиномиальной аппроксимацией; графики сохраняются в `experiments/demo/`.

Установка и запуск из корня проекта (для закреплённых зависимостей используйте Python 3.13):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter notebook experiments/
python -m unittest discover -s tests -v
```

В ноутбуке выберите ядро с установленными зависимостями и выполните **Restart & Run All**.
Пути к данным определяются относительно проекта, а не рабочего каталога.
При отсутствии MNE sample загрузчик скачает его в `data/` (около 2,7 ГБ после распаковки).
Импорт модулей из корня: `from src.cpop_segmentation import segment_cpop`,
`from utils.data_loading import load_xlsx_series`.
