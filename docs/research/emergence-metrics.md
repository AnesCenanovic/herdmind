# Research: Emergence Metrics (backlog item 14)
Status: first pass, 2026-11-10. Defines the four world-level emergence metrics
for item 16 (src/herdmind/eval/), directly aligned with EventRecord in
src/herdmind/world/log.py, BoardState in src/herdmind/world/ledger.py,
predation laws in src/herdmind/world/laws/predation.py, and upcoming
signalling (item 6).
## Summary
- All four metrics can be computed offline from a run folder using events.jsonl
(EventRecord, src/herdmind/world/log.py) and read_ledger(run_dir)
(list[BoardState], src/herdmind/world/ledger.py).
- Pack-hunting rate reads predation events (kind == "death",
cause == "eaten") from events.jsonl, locates the killing wolf (other_id)
in boards[step], and checks whether at least pack_size wolves are within
radius = 1 (predation.py).
- Herd cohesion uses the Vicsek polarization order parameter over per-step
sheep displacement vectors \(\Delta \mathbf{x}_i(t)\) tracked by agent_id,
paired with normalized nearest-neighbour distance (NND).
- Alarm-call survival effect measures the survival lift \(\Delta S_{\text{call}}\)
over horizon τ for threatened sheep exposed to a call within radius
\(R_{\text{call}}\) versus unexposed threatened sheep (requires item 6).
- Population cycle period extracts the dominant oscillation period T from
[b.count("sheep") for b in boards] after burn-in via FFT power spectral
density, paired with sheep-wolf cross-correlation lag.
## 1. Pack-hunting rate
### Intuition
In src/herdmind/world/laws/predation.py, the pack law requires at least
world.spec.laws.pack_size wolves within radius = 1 of the prey
(count_near(world, "wolf", prey.x, prey.y, radius=1)). Under always_kill or
chance laws, a single wolf can kill alone, so measuring how often multiple
wolves are present at the kill site tests whether pack hunting emerges without
being hard-enforced.
### Definition
Let \(\mathcal{K}\) be the set of predation events in events.jsonl after
burn-in t₀, filtered by kind == "death" and cause == "eaten". For
each event \(e \in \mathcal{K}\) at step = t, the killing wolf has
agent_id = e.other_id and stands on the prey's cell \((x_e, y_e)\) in
boards[t] (since predation occurs when a wolf ends its move on the sheep's
cell). Let w(e) be the number of wolves in boards[t] within radius
\(r_{\text{pack}} = 1\) of \((x_e, y_e)\) (matching predation.py):
\(R_{\text{pack}}=\frac{1}{|{}\mathcal{K}|{}}\sum _{e\in \mathcal{K}}\mathbb{I}\big(w(e)\ge k_{\min }\big)\in [0,1]\)
where \(k_{\min} = \text{pack\_size}\) (default \(k_{\min} = 2\)).
- Edge case: If \(\vert{}\mathcal{K}\vert{} = 0\), return NaN so zero-predation runs
are not conflated with solo-hunting runs (0.0).
- Secondary diagnostic: Mean wolf count at kills,
\(\bar{w} = \frac{1}{\vert{}\mathcal{K}\vert{}}\sum_{e \in \mathcal{K}} w(e) \ge 1\).
### References
- Muro, C., Escobedo, R., Spector, L., & Coppinger, R. P. (2011). Wolf-pack
(Canis lupus) hunting strategies emerge from simple rules in computational
simulations. Behavioural Processes, 88(3), 192-197.
- Escobedo, R., Muro, C., Spector, L., & Coppinger, R. P. (2014). Group size,
individual role differentiation and effectiveness of cooperation in a
homogeneous group of hunters. Journal of the Royal Society Interface, 11(95).
## 2. Herd cohesion (order parameter)
### Intuition
Herding on a grid combines directional alignment (sheep moving together in
the same direction) and spatial clustering (sheep staying close together).
### Definition
Match sheep across consecutive snapshots boards[t - 1] and boards[t] by
agent_id. Let \(\mathcal{S}_{t}\) be the set of sheep alive at both t - 1 and
t with \(N_t = \vert{}\mathcal{S}_t\vert{}\). For each \(i \in \mathcal{S}_t\), let
\(\Delta \mathbf{x}_i(t) = (x_i(t) - x_i(t-1),\, y_i(t) - y_i(t-1))\) be its step
displacement (using shortest torus delta if the world wraps) and define the unit
heading vector:
\(\^{\mathbf{v}}_{i}(t)=\begin{cases}\dfrac{\Delta \mathbf{x}_{i}(t)}{\|{}\Delta \mathbf{x}_{i}(t)\|{}_{2}},&\text{if\ }\|{}\Delta \mathbf{x}_{i}(t)\|{}_{2}>0\\ \mathbf{0},&\text{if\ }\|{}\Delta \mathbf{x}_{i}(t)\|{}_{2}=0\end{cases}\)
The Vicsek polarization order parameter at step t is:
\(\Phi _{t}=\frac{1}{N_{t}}\left\|{}\sum _{i\in \mathcal{S}_{t}}\^{\mathbf{v}}_{i}(t)\right\|{}_{2}\in [0,1]\)
Run-level cohesion is the post-burn-in mean \(\bar{\Phi} = \frac{1}{T - t_0}\sum_{t=t_0+1}^{T} \Phi_t\).
- Interpretation: \(\Phi_t = 1\) when all sheep move in the exact same
direction; \(\Phi_t \approx 0\) when movement is uncorrelated or all sheep are
stationary.
- Edge case: If \(N_t < 2\), \(\Phi_t = \text{NaN}\) and the step is excluded
from Φ̄.
- Companion spatial metric (normalized NND): Because stationary sheep yield
\(\Phi_t = 0\) even when tightly grouped, pair \(\Phi _{t}\) with mean
nearest-neighbour distance normalized by expected random spacing on a grid of
area A = H × W:
\(\~{d}_{\text{NN}}(t)=\frac{1}{N_{t}}\sum _{i\in \mathcal{S}_{t}}\min _{j\ne i}d(i,j)\Big/\left(\frac{\sqrt{A}}{2\sqrt{N_{t}}}\right)\)
Values \(\tilde{d}_{\text{NN}} < 1\) indicate spatial aggregation; \(\tilde{d}_{\text{NN}} \approx 1\)
indicates random dispersion.
### References
- Vicsek, T., Czirók, A., Ben-Jacob, E., Cohen, I., & Shochet, O. (1995). Novel
type of phase transition in a system of self-driven particles. Physical Review
Letters, 75(6), 1226-1229.
- Couzin, I. D., Krause, J., James, R., Ruxton, G. D., & Franks, N. R. (2002).
Collective memory and spatial sorting in animal groups. Journal of Theoretical
Biology, 218(1), 1-11.
## 3. Alarm-call survival effect
### Intuition
Once item 6 adds the call action (logged either in decisions.jsonl where
chosen == "call" or as an EventRecord(kind="call") in events.jsonl) and
auditory radius \(R_{\text{call}}\), a sheep that spots a wolf can emit an
alarm call. To test whether signalling has a real survival advantage under
predation threat, compare survival rates of threatened sheep that heard a
recent call against threatened sheep that did not.
### Definition
Restrict evaluation to threatened sheep-step observations (i, t) where sheep
i is alive in boards[t] and at least one wolf is within threat radius
\(R_{\text{threat}}\) (e.g. wolf perception/pursuit radius). Partition into:
- Exposed (\(\mathcal{E}_{1}\)): At least one other sheep j ≠ i emitted
call within \(d(i, j) \le R_{\text{call}}\) during \([t - W,\, t]\) (default
W = 3 steps).
- Unexposed (\(\mathcal{E}_{0}\)): No call received within \(R_{\text{call}}\)
during \([t - W,\, t]\).
Let \(Y_{i,t}(\tau) \in \{0, 1\}\) indicate whether sheep i (agent_id) is
still present in boards[t + \tau] (default horizon τ = 5 steps, excluding
right-censored steps t > T - τ). The alarm-call survival effect is:
\(\Delta S_{\text{call}}(\tau )=\frac{1}{|{}\mathcal{E}_{1}|{}}\sum _{(i,t)\in \mathcal{E}_{1}}Y_{i,t}(\tau )-\frac{1}{|{}\mathcal{E}_{0}|{}}\sum _{(i,t)\in \mathcal{E}_{0}}Y_{i,t}(\tau )\in [-1,1]\)
- Interpretation: \(\Delta S_{\text{call}} > 0\) means hearing an alarm call
increases survival probability under threat; \(\Delta S_{\text{call}} \approx 0\)
means calls have no measurable protective effect.
- Edge case: Before item 6 is implemented (or when \(\vert{}\mathcal{E}_1\vert{} = 0\) or
\(\vert{}\mathcal{E}_0\vert{} = 0\)), return NaN.
### References
- Maynard Smith, J., & Harper, D. (2003). Animal Signals. Oxford University Press.
- Ackley, D. H., & Littman, M. S. (1994). Altruism in the evolution of
communication. In Artificial Life IV (pp. 40-48). MIT Press.
## 4. Population cycle period
### Intuition
Balanced predator-prey dynamics produce sustained oscillations (Lotka-Volterra
cycles), whereas poorly calibrated models collapse to early extinction or
static equilibria.
### Definition
Extract \(N_{\text{sheep}}(t) = \text{boards}[t].\text{count}(\text{"sheep"})\) and
\(N_{\text{wolf}}(t) = \text{boards}[t].\text{count}(\text{"wolf"})\) for
\(t = t_0 + 1, \dots, T\) after discarding burn-in \(t_0 = \lfloor 0.1\,T \rfloor\).
1. Demean: \(x(t) = N_{\text{sheep}}(t) - \bar{N}_{\text{sheep}}\) over
M = T - t₀ steps.
2. Power Spectral Density (FFT): Compute one-sided power at frequencies
\(f_k = k / M\) for \(k = 1, \dots, \lfloor M/2 \rfloor\):
\(P(f_{k})=\frac{1}{M}|{}X(f_{k})|{}^{2}\)
3. Dominant cycle period: With \(k^* = \arg\max_{k \ge 1} P(f_k)\):
\(T_{\text{cycle}}=\frac{1}{f_{k^{*}}}=\frac{M}{k^{*}}\text{\ steps}\)
Report alongside three companion diagnostics:
- Spectral purity: \(\rho_{\text{spec}} = P(f_{k^*}) / \sum_{k \ge 1} P(f_k) \in (0, 1]\),
measuring how strongly power concentrates in a single cycle frequency.
- Predator-prey phase lag (\(\Delta t^*\)): Positive lag \(\ell > 0\) maximizing
normalized cross-correlation between demeaned \(N_{\text{sheep}}(t)\) and
\(N_{\text{wolf}}(t + \ell)\) (verifying that wolf peaks lag behind sheep peaks).
- Extinction indicator & step: Whether \(N_{\text{sheep}}(T) = 0\) or
\(N_{\text{wolf}}(T) = 0\), and the first step \(t_{\text{ext}}\) at which
either species hits zero.
### References
- Lotka, A. J. (1925). Elements of Physical Biology. Williams & Wilkins.
- Volterra, V. (1926). Fluctuations in the abundance of a species considered
mathematically. Nature, 118, 558-560.
- Wilensky, U. (1997). NetLogo Wolf Sheep Predation model. Center for Connected
Learning and Computer-Based Modeling, Northwestern University.
## Summary table for item 16

Metric	Symbol	Range	Primary source in world/	NaN condition
Pack-hunting rate	\(R_{\text{pack}}\)	\([0, 1]\)	events.jsonl (cause="eaten") + BoardState	\(\vert{}\mathcal{K}\vert{} = 0\)
Herd cohesion	Φ̄	\([0, 1]\)	Consecutive BoardState.animals by agent_id	\(N_t < 2\)
Alarm-call survival effect	\(\Delta S_{\text{call}}\)	\([-1, 1]\)	BoardState.animals + call log (item 6)	\(\vert{}\mathcal{E}_1\vert{} = 0\) or \(\vert{}\mathcal{E}_0\vert{} = 0\)
Population cycle period	\(T_{\text{cycle}}\)	\([2, M]\) steps	BoardState.count("sheep")	Extinction before t₀ + 2

## Open questions before item 16
1. Logging call events in item 6: Confirm whether item 6 will emit
EventRecord(kind="call", ...) in events.jsonl or whether item 16 should
read chosen == "call" from decisions.jsonl joined with boards[step].
2. Spectral purity cutoff: Decide whether runs with \(\rho_{\text{spec}} < 0.15\)
(flat or noisy trajectories) should return T_cycle = NaN or report the raw
argmax period alongside \(\rho _{\text{spec}}\).