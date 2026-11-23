## 1. Постановка задачи

Игра идёт на неограниченной плоскости. **Преследователь (шофёр)** стремится как можно раньше поймать пешехода. Его скорость постоянна, а радиус поворота ограничен снизу. В единицах модели скорость шофёра и минимальный радиус поворота равны единице. **Убегающий (пешеход)** стремится избежать поимки или отсрочить её. Он движется медленнее, с постоянной скоростью $\gamma=v_E/v_P$, но может мгновенно менять направление бега.[^game]

Состояние $(x,y)$ задаёт положение пешехода относительно шофёра. Ось $y$ направлена вперёд по курсу автомобиля, ось $x$ — вправо. Шофёр выбирает угловую скорость $\phi\in[-1,1]$. Положительный знак означает поворот вправо, отрицательный — влево, ноль — движение прямо. Пешеход выбирает курс $\psi\in[-\pi,\pi]$, отсчитываемый по часовой стрелке от направления движения шофёра. Относительное движение описывается следующей системой.[^game]

$$
\begin{aligned}
\dot x &= -\phi y + \gamma\sin\psi,\\
\dot y &= -1 + \phi x + \gamma\cos\psi.
\end{aligned}
$$

**Поимка** происходит при первом контакте с замкнутым кругом $\sqrt{x^2+y^2}\leq\beta$, включая касание между моментами выбора управления. В проекте $\gamma=0.5$, $\beta=0.3$, горизонт $T=20$. Контакт ровно в $T$ засчитывается шофёру. Если к моменту $T$ контакта не было, пешеход успешно уклонился на этом горизонте. Если для одного из игроков не удалось вычислить действие, исход помечается как неопределённый (`unsupported`).[^capture]

**Критерий сравнения шофёров** — доля стартов, из которых шофёр поймал одного и того же классического пешехода за отведённое время. Для обоих шофёров берут одни и те же 6 532 точки сетки с шагом $0.1$ в квадрате $[-4,4]^2$ вне круга поимки. Квадрат ограничивает только старт. Попарную разницу времени поимки считают по стартам, где захвата достигли оба шофёра.[^eval]

## 2. Стратегии управления

**Классический убегающий.** По текущим $(x,y)$ и параметрам игры пешеход определяет, какая дуга классического решения Мерца проходит через его положение. Курс $\psi$ берётся из геометрии этой дуги. Например, на дуге выхода к положительной полуоси стратегия находит оставшийся угол поворота шофёра и по нему задаёт курс пешехода. Когда пешеход уже прямо впереди ($x=0$, $y>\beta$), он бежит по ходу машины, $\psi=0$. На обычной дуге курс задан даже при отклонении шофёра от классического манёвра. На линии рассеяния, где возможны два продолжения, пешеход выбирает между двумя курсами по знаку **фактически применённого** руля шофёра.[^evader]

**Классический преследователь.** Шофёр использует те же $(x,y)$ и геометрию Мерца. По положению цели относительно барьера и линий переключения он выбирает манёвр. На быстрой стороне барьера шофёр поворачивает к цели с предельной угловой скоростью. На другой стороне возможен предельный *отворот* от цели, например для выхода на отрицательную полуось. Когда пешеход оказывается прямо впереди, шофёр продолжает погоню по прямой ($\phi=0$). Для симметричной позиции слева знак поворота меняется.[^driver]

**Эвристический преследователь.** Шофёру достаточно текущего направления на пешехода и длительности шага $\Delta t=0.05$. Он вычисляет угол до цели $\alpha=\operatorname{atan2}(x,y)$ и запрашивает угловую скорость, которая за один шаг повернула бы машину на этот угол. Результат ограничивается допустимым диапазоном. Если цель строго сзади, выбирается $\alpha=\pi$ и поворот вправо.[^heur]

$$
\phi=\max\!\left(-1,\min\!\left(1,\frac{\alpha}{\Delta t}\right)\right).
$$

Различие видно в контрольной точке P4, $(x,y)=(0.45,1)$. Пешеход впереди справа. Эвристика сразу поворачивает к нему вправо ($\phi=+1$), а классический шофёр начинает с левого поворота ($\phi=-1$), чтобы выйти на ось позади машины.[^p4]

### Неясные сведения и границы применимости

- **Какой курс постоянен между решениями?** В постановке Notion описан базовый шаг Рунге — Кутты с постоянными $\phi$ и относительным курсом $\psi$. В протоколе сравнения постоянны руль и **абсолютный** курс пешехода, поэтому относительный курс меняется по закону $\dot\psi=-\phi$. Оба игрока выбирают действия заново после события переключения или по окончании шага.[^step]
- **Где подтверждена применимость классики?** Реализация классической обратной связи принимает только $\gamma=0.5$ и $\beta=0.3$. Сравнение на конечном горизонте оценивает шофёра против фиксированного алгоритма пешехода. Из него нельзя установить время поимки при наилучшем ответе пешехода на произвольную стратегию шофёра или сделать вывод о вечном уклонении при тайм-ауте.[^scope]

[^game]: [Notion, «Постановка задачи», раздел «Описание»](https://app.notion.com/p/4122971d9e9980b88422ce4b334f05b9) и [«Классические стратегии», §1–2](https://app.notion.com/p/4122971d9e99801991fef28883d7db17). Код — [homicidal_chauffeur.py, `GameConfig`, `dynamics`, строки 23–49](../chauffeur_evader/homicidal_chauffeur.py#L23-L49) и [`HomicidalChauffeurEnv.step_controls`, строки 257–269](../chauffeur_evader/homicidal_chauffeur.py#L257-L269).
[^capture]: [homicidal_chauffeur.py, `GameConfig` и `is_captured`, строки 23–54](../chauffeur_evader/homicidal_chauffeur.py#L23-L54), [`HomicidalChauffeurEnv._finish_step`, строки 301–321](../chauffeur_evader/homicidal_chauffeur.py#L301-L321). В протоколе сравнения — [chauffeur_classical.py, `_capture_time`, строки 409–433](../chauffeur_evader/chauffeur_classical.py#L409-L433), [`step_feedback`, строки 551–565](../chauffeur_evader/chauffeur_classical.py#L551-L565) и [656–681](../chauffeur_evader/chauffeur_classical.py#L656-L681).
[^eval]: [Notion, задача 06, «Метрики»](https://app.notion.com/p/4112971d9e9981fbb77ffdb6e3beef28) и [итоговый протокол, «Итоги на сетке»](https://app.notion.com/p/4192971d9e99814eb584d068264f3220). Код — [chauffeur_evaluation.py, `final_grid`, строки 31–46](../chauffeur_evader/chauffeur_evaluation.py#L31-L46), [`evaluate_pursuer`, строки 75–89](../chauffeur_evader/chauffeur_evaluation.py#L75-L89), [`_outcome_summary` и `compare`, строки 126–148](../chauffeur_evader/chauffeur_evaluation.py#L126-L148).
[^evader]: [Notion, «Классические стратегии», §6–8](https://app.notion.com/p/4122971d9e99801991fef28883d7db17). Код — [chauffeur_classical.py, `_upper`, строки 73–79](../chauffeur_evader/chauffeur_classical.py#L73-L79), [`classical_feedback` на оси, строки 285–299](../chauffeur_evader/chauffeur_classical.py#L285-L299), [`Feedback.response`, строки 52–65](../chauffeur_evader/chauffeur_classical.py#L52-L65), [`classical_evader`, строки 370–376](../chauffeur_evader/chauffeur_classical.py#L370-L376) и [`step_feedback`, строки 551–562](../chauffeur_evader/chauffeur_classical.py#L551-L562).
[^driver]: [Notion, «Классические стратегии», §6–8](https://app.notion.com/p/4122971d9e99801991fef28883d7db17). Код — [chauffeur_classical.py, `_primary`, строки 82–105](../chauffeur_evader/chauffeur_classical.py#L82-L105), [`_away`, строки 129–149](../chauffeur_evader/chauffeur_classical.py#L129-L149), [`classical_feedback` на оси, строки 285–292](../chauffeur_evader/chauffeur_classical.py#L285-L292), [выбор ветви, отражение и `classical_pursuer`, строки 339–367](../chauffeur_evader/chauffeur_classical.py#L339-L367).
[^heur]: [chauffeur_evaluation.py, `heuristic_pursuer`, строки 36–40](../chauffeur_evader/chauffeur_evaluation.py#L36-L40), [homicidal_chauffeur.py, `GameConfig.dt`, строки 23–30](../chauffeur_evader/homicidal_chauffeur.py#L23-L30).
[^p4]: [tests/test_chauffeur_classical.py, `test_p2_through_p8_from_the_report_and_their_reflections`, строки 129–151](../tests/test_chauffeur_classical.py#L129-L151), [chauffeur_evaluation.py, `heuristic_pursuer`, строки 36–40](../chauffeur_evader/chauffeur_evaluation.py#L36-L40) и [Notion, «Как читать расхождения», P4](https://app.notion.com/p/4192971d9e99814eb584d068264f3220).
[^step]: [Notion, «Постановка задачи», раздел «Параметры среды»](https://app.notion.com/p/4122971d9e9980b88422ce4b334f05b9). Базовый шаг — [homicidal_chauffeur.py, `_rk4_path`, строки 57–71](../chauffeur_evader/homicidal_chauffeur.py#L57-L71). Шаг сравнения — [chauffeur_classical.py, `_path` и `_capture_time`, строки 396–419](../chauffeur_evader/chauffeur_classical.py#L396-L419), [`step_feedback`, строки 631–663](../chauffeur_evader/chauffeur_classical.py#L631-L663) и [`rollout_feedback`, строки 705–737](../chauffeur_evader/chauffeur_classical.py#L705-L737).
[^scope]: [chauffeur_theory.py, константы `GAMMA`, `BETA`, строки 18–20](../chauffeur_evader/chauffeur_theory.py#L18-L20), [chauffeur_classical.py, `_check_parameters`, строки 68–70](../chauffeur_evader/chauffeur_classical.py#L68-L70), [chauffeur_evaluation.py, `evaluate_pursuer`, строки 75–89](../chauffeur_evader/chauffeur_evaluation.py#L75-L89) и [Notion, «Как читать расхождения»](https://app.notion.com/p/4192971d9e99814eb584d068264f3220).
