# Global-buffer architecture

## Processing path

1. A stateless classifier produces the task type and semantic features.
2. LVP-style load balancing selects exactly one responsible controller.
3. That controller evaluates every eligible task--agent pair and publishes an immutable
   base score and predicted processing/quality/waiting values.
4. The global buffer refreshes only time and availability values.
5. At every arrival or completion event, the buffer matches distinct pending
   tasks to distinct free agents.
6. An agent receives no new task until its current task completes.
7. The judge evaluates the resulting request--response pair.
8. Feedback updates only the controller that published that task.
9. Local RLS updates run after each completion; A-SPSA periodically
   synchronizes processing-time, waiting-time, and quality predictors.

## Controller score and buffer priority

For task `i`, agent `k`, and responsible controller `j`, the controller creates
a feature vector `f_ik` and immutable base score

```text
s_jik = w_j^T f_ik.
```

The buffer computes

```text
finish_ik(t) = max(t, available_k(t)) + p_hat_ik
d_ik(t)      = max(0, finish_ik(t) - D_i) / b_i
r_ik(t)      = (max(t, available_k(t)) - arrival_i + p_hat_ik) / b_i
K_ik(t)      = (D_i - max(t, available_k(t))) / (p_hat_ik + epsilon)
```

The appendix of the manuscript writes `finish_ik = available_k + p_hat_ik`.
The implementation clamps the start to the current time, because a task cannot
begin before the scheduler wakes; for a free agent the two forms differ.

The task-level hotness uses the best slack among the agents this dispatch round
can actually reach:

```text
H_i(t) = sigmoid(alpha * (1 - max_{k free} K_ik(t))).
```

Restricting the maximum to reachable agents matters: taking it over every
agent, including occupied ones, reported a comfortable task whenever some busy
agent would eventually have slack, even when every option available now was
already past its deadline.

Waiting pressure is also task-level:

```text
W_i(t) = min((t - arrival_i) / w_hat_i, cap).
```

`w_hat_i` is the controller's published admissible-wait prediction. It enters
the dispatch order only. It never defines the external deadline `D_i` or the
time budget `b_i`, so the waiting model cannot define the metric it is scored
on, yet improving it can still change p95 wait and deadline-hit rate.

The scheduling priority is

```text
P_jik(t) = s_jik
         + lambda_H * H_i(t)
         + lambda_W * W_i(t)
         - lambda_D * d_ik(t)
         - lambda_R * max(0, r_ik(t) - 1).
```

There is deliberately no availability term. Dispatch considers agents with
`available_k(t) <= t` only, so `availability_ik(t) = exp(0) = 1` for every
candidate edge of a dispatch round. The term added the same constant to every
edge and could never change a matching, in either the greedy or the exact
matcher. `availability_score` is still refreshed and kept as an audit field.

Hotness and waiting pressure are task-level, so they cannot make an unsuitable
agent win; they raise a task in the global buffer without changing which agent
is most suitable for it.

## Matching invariants

- A task has one responsible controller.
- The controller publishes one unique proposal for every eligible agent.
- A task is assigned to at most one agent.
- A free agent is assigned at most one task per dispatch round.
- Alternatives are retained for audit but marked `INVALIDATED` after dispatch.
- An execution failure is terminal and produces negative feedback; there is no
  implicit retry.
- Pending base scores are never recomputed after publication.

Exact maximum-weight bipartite matching is the primary dispatch strategy;
deterministic greedy matching is available as a baseline and ablation. The
buffer keeps a live pending set, so an event costs work proportional to the
tasks still waiting rather than to every task ever submitted.

## Adaptation

Each controller owns local routing weights `w_j`. Only the responsible
controller updates them from observed routing utility. The implementation uses
a projected online update with a small anchor toward the initial policy.

A-SPSA does not own the final dispatch decision. It synchronizes controller
prediction models after completed feedback has accumulated. Routing weights
remain local so controllers can specialize.

The consensus correction shared by every compared optimizer is
`omega * sum_l b_jl (theta_j - theta_l)`, the exact gradient of the
disagreement term of the network objective `Phi`. It is added to the descent
direction, so the realised consensus strength is `omega * h` for every method.
See `PAPER_EXPERIMENTS.md` for why this had to be made uniform.

## Time model

The simulator is discrete-event rather than fixed-step polling. Task arrivals
may be batched at the same timestamp, and service durations may be expressed in
integer ticks for numerical experiments. The scheduler wakes on an arrival or
agent completion. Real deployments use the same event transitions with
continuous wall-clock timestamps.

Offered load `rho` is normalized against a fixed canonical six-agent capacity
so it does not change when a candidate architecture changes its agent count. A
larger system therefore runs cooler than `rho` suggests; every run summary also
reports `rho_effective` and `mean_utilization`.
