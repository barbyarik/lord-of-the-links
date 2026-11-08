# Запуск и входы

## Веса

Передайте путь к папке через `--models`. Имена файлов нужны для загрузчиков. Для воспроизведения можно сверить SHA-256 с таблицей.

| Путь в папке моделей | Версия и источник | SHA-256 |
| --- | --- | --- |
| `yolo11m.pt` | [YOLO11m, Ultralytics Assets v8.3.0](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt) | `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` |
| `clip-reid-msmt17.pth` | Строка ViT-CLIP-ReID, колонка MSMT17 в [таблице авторов](https://github.com/Syliz517/CLIP-ReID#trained-models-and-test-logs) | `9974ef57c418e8c88e59581f66f3f53fbe02f879311f7438009f84fd150b1c63` |
| `grounding-dino-tiny/` | Конфиг, токенизатор, препроцессор и веса из [репозитория модели](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/main) | `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3` для `model.safetensors` |

Исходные видео для инференса — пять MKV, предоставленных организаторами. Путь к каждому указывается отдельно, как в README. Готовые демо содержат MP4 для просмотра. При повторном инференсе на предоставленной записи используйте исходные MKV, чтобы сохранить их PTS и номера кадров.

`requirements.txt` закрепляет версии библиотек. Установку и запуск проверяли на macOS arm64 с Python 3.14. `--device cpu` подходит для проверки и повтора из кэшей. Для полного инференса можно выбрать `mps` или `cuda:0` в окружении, где PyTorch видит это устройство.

## Оценить и собрать интерфейс

```sh
python -m src evaluate --predictions dist/predictions.json \
  --reference artifacts/reference --end 1791482323 --output dist/metrics.json

python -m src build-demo --predictions dist/predictions.json --output dist/demo \
  --video 1=.local/videos/camera_new_1.mkv --video 2=.local/videos/camera_new_2.mkv \
  --video 3=.local/videos/camera_new_3.mkv --video 4=.local/videos/camera_new_4.mkv \
  --video 5=.local/videos/camera_new_5.mkv

python -m src.web.server --dataset dist/demo
```

Экспорт сохраняет сетку наблюдений 5 Гц и исходный ход времени у видео. Сглаживание с `σ = 0,32 с` применяется к позициям для карты. Оно не меняет боксы, глобальные ID и метрики.

Отдельный `lol-demo-reference.zip` показывает ручную разметку на её сетке 2 Гц. Для численного сравнения используется версия эталона в `artifacts/reference` и проверенные кадры на сетке 1 Гц. [Методика оценки](EXPERIMENTS.md)

## Повторить сшивку из кэша

Распакуйте `lol-replay-cache.zip` в `.local/replay`.

```sh
python -m src run --profile balanced --recording config/recordings/hackathon.json \
  --detections .local/replay/detections --appearance .local/replay/appearance.npz \
  --scene-cache .local/replay/scene.json --output dist/balanced.json
```

Кэш детекций хранит сырые рамки YOLO. Кэш внешности хранит векторы точных кропов. Кэш сцены содержит автоматически найденные экраны, зеркала и мебель. Команда повторно запускает ByteTrack и сшивку. Она может менять правила и профиль на тех же видео без повторного инференса моделей. Для свежего прогона флаги `--save-detections` и `--save-appearance` сохраняют новые кэши. Если передать и видео, и кэш, сверяется SHA-256 источника.

Профиль — JSON со списком включённых правил. Готовые пресеты находятся в `config/profiles/`. Копию профиля со своими флагами можно передать через `--profile путь/к/профилю.json`. Порядок стадий задаёт код пайплайна. [Правила по стадиям](ARCHITECTURE.md)

## Подключить новый день или офис

Для нового дня на тех же камерах сохраните калибровку и задайте свой JSON записи. Укажите `recording_start`, при необходимости `recording_end`, и для каждой камеры либо `clock_anchors` (пары `[source_pts_s, common_time_s]`), либо `offset_s` относительно начала. Для другого дня запустите модели по новым видео.

Для нового офиса задайте собственный план, граф камер и допустимые зоны. Сопоставьте на каждой камере минимум четыре разнесённые точки одной плоскости. Например, углы дверных порогов и основания стен. `python -m src.calibrate --help` показывает формат расчёта матриц из неподвижных ориентиров. Передайте новый конфиг трекеру через `--site`, а калибровку и план экспортёру через `--calibration` и `--plan`. Для воспроизведения этого офиса сохраняются исторические матрицы, две из которых уточнялись по синхронным проходам. [Геометрия](ARCHITECTURE.md)
