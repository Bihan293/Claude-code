# opus-agent — автономный coding agent для Android Termux

Аналог Claude Code для терминала Termux. Работает напрямую с Anthropic Messages API
(Tooken Club, `claude-opus-5-5`). Агент сам проходит цикл
**анализ → план → код → тесты → исправления → commit → push → PR → CI**.

```
$ cd ~/projects/my-repo
$ opus
> Изучи этот репозиторий, реализуй X, добавь тесты, запусти проверки, исправь проблемы и создай PR
```

## Установка (Termux)

```bash
pkg install -y git python
git clone https://github.com/Bihan293/Claude-code.git ~/.opus-agent-src
cd ~/.opus-agent-src && bash install.sh
opus setup          # спросит Base URL, модель и API key (ввод скрыт), проверит ключ
```

Скрипт ставит `python git ripgrep gh termux-api` и пакет `opus-agent` (зависимости: `httpx`, `rich`,
`prompt_toolkit`, все pure-python). Для уведомлений Android и wake-lock поставьте приложение **Termux:API**.

## Секреты

* Ключи хранятся в `~/.opus-agent/credentials.json` с правами `600`, отдельно от `config.json`.
* Ключ можно передать и через env: `TOOKEN_API_KEY`, `GITHUB_TOKEN`, `TELEGRAM_BOT_TOKEN`.
* Все логи, вывод инструментов и память проходят через редактор секретов: зарегистрированные ключи и
  типовые токены (`sk-…`, `ghp_…`, `github_pat_…`, Telegram, `user:pass@` в URL) заменяются на `***REDACTED***`.
* Дочерние процессы (shell агента) **не получают** API-ключ LLM. GitHub-токен передаётся git через
  `GIT_ASKPASS`, а `gh` через env. В URL remote и `.git/config` токен не попадает.

## Использование

| Команда | Что делает |
|---|---|
| `opus` | интерактивный REPL в текущей папке |
| `opus "задача"` | REPL, сразу начинает с задачи |
| `opus -p "задача"` | headless: выполняет задачу автономно, печатает отчёт, выходит |
| `opus --detach "задача"` | то же в фоне (можно закрыть REPL), лог в `~/.opus-agent/logs/run-*.log` |
| `opus -c` | продолжить последнюю сессию проекта (после обрыва, OOM, перезагрузки) |
| `opus -r <id>` | продолжить конкретную сессию |
| `opus --mode plan` | режим только для чтения: агент исследует проект и выдаёт план |
| `opus --mode ask` | спрашивать подтверждение на запись и shell |
| `opus setup` | настройка API |
| `opus github login\|status\|logout` | подключение GitHub (PAT) |
| `opus clone owner/repo` | клонировать в `~/projects/repo` |
| `opus projects` | список проектов |
| `opus sessions` / `opus status` | сессии / состояние последней задачи и фонового запуска |
| `opus usage [дни]` | расход токенов по дням, доля попаданий в кэш, примерная стоимость |
| `opus memory show\|edit\|clear [global]` | долговременная память |
| `opus telegram setup\|test\|on\|off` | уведомления в Telegram |
| `opus config [key [value]]` | показать или изменить настройки |
| `opus doctor` | проверка окружения и API |

### Slash-команды в REPL

`/status /usage /todos /compact /clear /continue /sessions /resume <id> /mode auto|ask|plan /model
/thinking adaptive|enabled|off /undo [n] /diff /cd /projects /clone /memory /remember <текст>
/init /issue <n> /pr /review [n] /fixci /verbose /exit`

* `/issue 42` — реализовать issue от начала до конца и открыть PR с `Closes #42`.
* `/fixci` — проверить CI, скачать логи упавших jobs, исправить, сделать push, повторять до зелёного.
* `/init` — изучить репозиторий и записать `OPUS.md` с инструкциями по проекту.
* `Ctrl+C` — прервать агента. Состояние сохранено, `/continue` продолжает работу.

## Возможности агента (инструменты)

| Инструмент | Назначение |
|---|---|
| `read_file` `write_file` `edit_file` `multi_edit` `delete_path` `move_path` | файлы (точечные правки, атомарный multi-edit, защита от редактирования устаревшей версии файла, изображения) |
| `list_dir` `glob` `grep` | навигация и поиск (ripgrep, если установлен) |
| `bash` `job_output` `job_kill` `job_list` | shell с таймаутами, фоновые процессы, cwd сохраняется между вызовами |
| `github` | repo_info, issues, PR (create/get/diff/comment/merge), checks/CI, логи workflow, rerun |
| `github_api` | любой вызов GitHub REST |
| `todo_write` | план задачи (сохраняется в сессии) |
| `memory` | долговременная память: проект и глобальная |
| `task` | sub-agent со свежим контекстом (explore = только чтение; general). Несколько запускаются параллельно |
| `web_fetch` `web_search` | документация и поиск |
| `ask_user` | вопрос пользователю (в автономном режиме отключён) |

Независимые read-only вызовы из одного ответа модели выполняются **параллельно**.

## Git/GitHub workflow

* Агент не коммитит в `main/master`: создаёт ветку `feat/…`/`fix/…`. Push в default branch заблокирован
  на уровне инструмента (настройка `allow_push_to_default_branch`).
* Commit-сообщения в стиле Conventional Commits, `git push -u origin <branch>`, затем PR через API.
  Если PR для ветки уже есть, он обновляется.
* После push агент проверяет checks и при падении скачивает логи упавших шагов, исправляет и делает push снова.
* В конце задачи агент пишет итоговый отчёт со ссылкой на PR. Ссылка дублируется в строке статуса,
  в `opus status` и в уведомлениях.

## Экономия токенов

1. **Prompt caching**: 4 breakpoint'а (tools → system → два последних user-сообщения). Статическая часть
   промпта не меняется между вызовами. Кэш-чтение стоит примерно 10% цены input. Долю попаданий в кэш
   показывает `opus usage`.
2. **Ограничение вывода в источнике**: файлы читаются с offset/limit, у вывода shell сохраняется хвост
   (там обычно ошибки), progress-бары схлопываются, ANSI-коды удаляются.
3. **Pruning с гистерезисом**: при заполнении контекста на 55% старые большие tool-результаты, thinking
   прошлых ходов и большие входы `write_file` заменяются заглушками. Это делается одним пакетом и
   повторяется только после заметного роста контекста, чтобы кэш сбрасывался редко.
4. **Compaction**: при заполнении на 78% (или при ошибке «prompt too long») модель одним вызовом пишет
   сводку для продолжения работы: задача, архитектура, что сделано, ветка/PR, следующие шаги. Работа
   продолжается от этой сводки.
5. **Sub-agents** исследуют большие репозитории в своём контексте и возвращают только короткий отчёт.
   `small_model` позволяет отдавать им более дешёвую модель.
6. **Память проекта** (команды сборки и тестов, архитектура) и журнал прошлых задач избавляют от
   повторного исследования проекта.
7. **Adaptive thinking**: модель сама решает, сколько думать. Если прокси не поддерживает параметр,
   клиент сам откатывается на `enabled` с бюджетом, затем на `off`.

## Надёжность

* SSE-стриминг; если прокси вернул обычный JSON, клиент сам переключается на non-stream.
* Повторы с экспоненциальной задержкой и `Retry-After` для 429/5xx/529/сетевых ошибок.
* При ответе 400 на необязательный параметр (thinking/effort/cache ttl) клиент отключает этот параметр и
  повторяет запрос.
* Если ответ упёрся в `max_tokens`, агент автоматически продолжает. Обрезанный JSON инструмента
  возвращается модели с подсказкой разбить запись на части.
* Сессия сохраняется атомарно после каждого шага. При продолжении история восстанавливается: каждому
  `tool_use` нужна пара `tool_result`, и для оборванных вызовов добавляются заглушки.
* Все изменения файлов сохраняются в checkpoints, `/undo` откатывает их даже вне git.
* Пока идёт задача, держится `termux-wake-lock`, чтобы Android не усыпил процесс.

## Память

* `~/.opus-agent/memory/global.md` — предпочтения пользователя и факты об окружении.
* `~/.opus-agent/memory/projects/<repo>-<hash>.md` — знания о проекте (агент пишет их сам через `memory`).
* `…journal.md` — краткие итоги последних задач в проекте.
* В контекст автоматически подхватываются `OPUS.md`, `CLAUDE.md`, `AGENTS.md`,
  `.github/copilot-instructions.md` из репозитория.

## Telegram (необязательно)

`opus telegram setup`: токен бота от @BotFather, затем отправьте боту `/start`. chat_id определится
автоматически. Уведомления приходят о завершении задачи (с отчётом и ссылкой на PR), создании PR,
ошибках и вопросах агента. Без Telegram агент работает полностью.

## Основные настройки (`opus config <key> <value>`)

| ключ | по умолчанию | |
|---|---|---|
| `base_url` / `model` | `https://tooken.club/v1` / `claude-opus-5-5` | |
| `small_model` | `""` | модель для sub-agents (пусто = основная) |
| `max_tokens` | 32000 | лимит ответа |
| `thinking` / `thinking_budget` | adaptive / 12000 | |
| `effort` | `""` | `low/medium/high/max`, если API поддерживает `output_config.effort` |
| `context_window` | 200000 | для 1M-контекста поставьте 1000000 |
| `prune_threshold` / `compact_threshold` | 0.55 / 0.78 | |
| `cache_ttl` | 5m | `1h` для долгих сессий с паузами |
| `permission_mode` | auto | auto / ask / readonly |
| `max_iterations` | 300 | вызовов модели на задачу |
| `auth_style` | both | `x-api-key` и/или `Authorization: Bearer` |
| `price_*` | | цены за 1M токенов для оценки стоимости |

## Разработка

```bash
pip install -e '.[dev]' && pytest -q
```

Тесты используют локальный mock-сервер Anthropic SSE и проверяют полный цикл агента, ретраи и
откат параметров, compaction, sub-agents, восстановление после обрыва, `/undo`, редактирование
секретов и блокировку push в main.

## Структура

```
opus_agent/
  cli.py          CLI, REPL, slash-команды, подкоманды, detach
  agent.py        agent loop, параллельные инструменты, sub-agents, управление контекстом
  llm.py          клиент Messages API: SSE, ретраи, кэш, thinking, учёт usage
  context.py      оценка токенов, pruning, compaction
  session.py      сохранение сессий и восстановление истории
  memory.py       долговременная память и инструкции проекта
  prompts.py      системный промпт (статическая часть отдельно от динамической)
  checkpoints.py  резервные копии файлов для /undo
  notify.py       Telegram, termux-notification, wake-lock
  security.py     редактирование секретов в логах
  usage.py        учёт токенов по дням
  setup_wizard.py мастер настройки
  tools/          files, shell, github, misc (todo/memory/web/task/ask), gitauth
```

CI: workflow GitHub Actions лежит в `ci/github-actions.yml`. Чтобы включить, скопируйте его в
`.github/workflows/ci.yml` (`mkdir -p .github/workflows && cp ci/github-actions.yml .github/workflows/ci.yml`).
