# MCQ source resolver v1.1: закрытие trust boundary

дата: 2026-08-08

статус на момент заморозки: private/opaque MCQ-входы, assets, alignment map и gold этим исправлением не читались. opaque batch не запускался. accuracy не вычислялась и не заявляется.

## зачем понадобилась v1.1

независимый аудит коммита `463ee9f` нашел две реальные проблемы до допуска к blind-запуску.

первая проблема была в доверии к самосогласованному bundle. v1 проверял внутренние projection SHA, но вызывающий код мог подать одновременно измененные inventory, key index и render manifest, пересчитать их внутренние хэши и получить структурно корректный альтернативный источник. это особенно критично для official key: подмена ответа вместе с пересчетом key-index projection могла пройти обычный loader.

вторая проблема была в прямом `execute_mcq_opaque_batch`. CLI-loader проверял идентификаторы, JSON-схему, prompt и image SHA, но публичный Python API принимал уже созданные dataclass-объекты. вручную созданный объект мог обойти часть проверок loader, а небезопасный `input_id` затем использовался в имени certificate-файла.

v1.1 закрывает обе проблемы до чтения opaque bundle.

## новый trust anchor

операционный resolver больше не считает структурно валидные JSON-файлы доверенными. функция `assert_frozen_mcq_bundle` сначала проверяет байты опубликованного v1 freeze manifest по встроенному SHA-256:

`5744488edc02e70e921bae9cddbae3d2f60448768a7fd02975a7dc9e5ccb04f7`

затем она пересчитывает и проверяет:

- projection самого freeze manifest: `134946a0087cd1f389f2904b187f41708b7bb4ae8899001b5311b668aee14c01`;
- файл inventory: `965e41673aea7df73fb03f98818c3ce3c8a1561873c9deece7e11be9a8b37dec`;
- inventory projection: `5f9e01678b2a3b7c14600dffadb06e0cce96212712835509d3bcfd1625b4fff3`;
- файл official key index: `1ef19fe8e56b0ba97307d8c22abe8a8c8d3a72baf42eb5f7a22eb069b6a5e8e0`;
- key-index projection: `9ca8672db13c5d6a6b05ee375bc540c4b4e5647f91cb8c54daa4839fdaa317ee`;
- файл render manifest: `a57c8869ba29a5f9362d9d536b5a78415e810a4bfbdc8a3b2848b19e29cdb458`;
- render projection: `709ecc38a36cfbdad33d8ea8bf80ebf4ad38f00fd650655a5afd518c0d2903aa`;
- список и метаданные всех 28 PNG: `85bdb83c27bf4e4cc5e236a8997ecfb6903403cb3f1846685432672fb4c202f9`;
- SHA и размер каждого из 28 PNG непосредственно с диска;
- SHA frozen source-build audit и frozen report.

точное содержимое freeze manifest является корнем доверия. поэтому нельзя заменить ключ, пересчитать новый key projection и приложить новый самосогласованный freeze: SHA нового freeze уже не совпадет со встроенным trust anchor.

`assert_frozen_mcq_objects` повторно вызывается внутри resolver, certificate verifier и batch executor. он заново строит projections из dataclass-объектов, сравнивает их с точными frozen projections и повторно хэширует page bytes. это защищает прямые Python-вызовы, которые не используют CLI.

## v1.1 code freeze и внешний pin

data bundle сам по себе не доказывает, что после аудита не изменили Python-код. поэтому runner и операционные команды adapter теперь требуют второй manifest — v1.1 code freeze — и два обязательных внешних аргумента: ожидаемый SHA-256 файла и ожидаемый projection SHA-256. до разбора opaque input они проверяют self-projection freeze и SHA/размер каждого перечисленного code file. run manifest сохраняет file SHA, freeze projection, combined code projection и число проверенных файлов.

ожидаемый SHA нельзя безопасно встроить в сам runner: freeze хэширует runner, и возник бы циклический self-hash. поэтому trust anchor передается командой из опубликованных после freeze pins и закрепляется тем же Git commit. runner не имеет fallback/default и не принимает непроверенный v1.1 manifest.

code freeze включает не только MCQ-файлы, но и критические transitive dependencies: `visual_coordinate_binding.py`, `official_ogm.py`, `contracts.py`, package `__init__.py`, `certificates.py`, `policy.py` и `source_first.py`. изменение SIFT/RANSAC, geometry checks или canonical SHA/JSON helpers поэтому меняет freeze и блокирует запуск со старыми pins.

## opaque input boundary

`run_mcq_opaque_batch` теперь выполняет действия в следующем порядке:

1. полностью аттестует frozen public source bundle;
2. только после успешной аттестации читает opaque JSONL;
3. проверяет точный набор и типы полей каждой строки;
4. проверяет frozen Turkish prompt grammar;
5. проверяет path safety, SHA и реальные bytes ровно одного изображения;
6. запрещает duplicate/case-colliding/reserved Windows `input_id`;
7. запрещает повтор одного observable `(prompt, image)` при этом разрешая одну страницу с разными номерами вопросов;
8. запускает только встроенный source resolver без публичной подмены resolver callback;
9. пишет результаты в staging и атомарно публикует output directory.

прямой `execute_mcq_opaque_batch` также не доверяет готовым объектам. он получает raw JSONL bytes и asset root, повторно парсит их и требует точного равенства supplied objects наблюдаемым данным. traversal ID, duplicate, prompt relabel и замена image bytes отвергаются до создания output directory.

run manifest v1.1 фиксирует raw input JSONL SHA-256, размер в байтах и canonical ordered-input projection. дополнительно он записывает точные SHA source bundle и факт проверки exact source objects до разбора входа. внешний `run_mcq_opaque_batch` отдельно гарантирует более сильный порядок: полный file-level bundle attestation выполняется до чтения JSONL с диска.

## certificate verifier

verifier теперь обязательно получает ожидаемые image bytes и сравнивает их SHA с `task_image_sha256` сертификата. certificate другого изображения не может быть переиспользован.

граница проверки описана явно: verifier повторяет decision logic, source/key binding и projection сертификата по уже записанным visual evidences. он не утверждает, что заново вычислил SIFT. свежее SIFT-сопоставление делает только `resolve_mcq_image_bytes` в момент выпуска сертификата.

## воспроизводимость public source

перед v1.1 freeze выполнены две независимые public-source проверки в точном runtime:

- Python 3.12.13;
- pdfplumber 0.11.9;
- NumPy 2.5.1;
- OpenCV 5.0.0;
- Poppler 26.05.0;
- `pdftoppm.exe` SHA-256 `742cbbd9a00931ad16c6618410bc40471375d639a45c61c1d86f3dcfc54b6388`.

inventory, official key index и source audit были заново построены из двух pinned official MEB PDF. все три файла совпали побайтно с frozen artifacts.

freeze-команда сама создает новый временный каталог и внутри своего процесса заново рендерит все 28 content pages через pinned `pdftoppm`. готовый заранее replica каталог ей передать нельзя. после рендера временный render manifest совпадает побайтно с frozen manifest, а каждый из 28 PNG совпадает по SHA-256, размеру и геометрии. это проверка воспроизводимости source render, а не holdout evaluation.

## adversarial suite

на заморозке v1.1 проходит 56 тестов. кроме исходных threshold, prompt, source-census, unsupported-open-response и certificate-tamper проверок добавлены атаки:

- измененный official answer с пересчитанными cell/key-index projections;
- измененный inventory marker с пересчитанным inventory projection;
- измененный render SHA с пересчитанным render projection;
- подмененный freeze manifest;
- попытка прочитать opaque input до source attestation;
- traversal/reserved/duplicate identifiers;
- prompt relabel после loader;
- замена image bytes/SHA после loader;
- certificate, предъявленный с bytes другого изображения;
- подмененный runtime code file внутри нового самосогласованного v1.1 freeze;
- плохой внешний v1.1 pin в adapter дает штатный exit code 2 без traceback и без чтения image;
- проверка raw JSONL SHA, размера и ordered projection в run manifest.

Python, OpenCV, NumPy и pdfplumber сейчас закреплены точными версиями и проверяются в runtime, но freeze не хэширует целиком их установленные distributions/все native binaries. это явно остается environment assumption уровня воспроизводимости. `pdftoppm.exe`, напротив, закреплен точным binary SHA. для максимально строгого production replay следующий инфраструктурный слой должен запускать commit в immutable container image с опубликованным image digest.

## честная интерпретация

v1.1 не повышает и не измеряет accuracy. это исправление целостности эксперимента. его результатом является возможность выполнить blind MCQ source lookup так, чтобы ответы нельзя было объяснить подменой official key, подменой source bundle, использованием alignment ID или обходом loader через прямой API.

четыре public Physics Unit 2 задания 24–27 остаются `unsupported_open_response`: исходный public protocol ошибочно относил их к A–E. resolver abstains и не создает для них искусственные ответы. выбран ли какой-либо из этих адресов в holdout, до завершения v1.1 freeze не проверялось.

дальнейший допустимый порядок: закоммитить v1.1 вместе с freeze manifest, получить независимый audit PASS, только затем открыть opaque prompt/image bundle, выполнить batch, запечатать outputs и лишь после этого открыть private alignment/gold для evaluation.
