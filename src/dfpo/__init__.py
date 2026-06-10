"""dfpo — Drone Fuzzy-PSO Optimizer.

Núcleo científico do projeto "Sistema Inteligente Fuzzy-Evolutivo de Otimização de Asas para Drones
de Entrega" (IA e Computacional, CESUPA, CC5NA).

Subpacotes:
    baseline    -- superfície analítica de referência f(x, y) (gradiente, Hessiana, ótimo).
    fuzzy       -- sistema fuzzy Mamdani (scikit-fuzzy): MFs, variáveis, regras, inferência.
    pso         -- Particle Swarm Optimization (implementado manualmente): swarm, calibrate, operate.
    reference   -- verdade de referência p/ calibração (superfície analítica + dataset sintético).
    experiments -- protocolo experimental: runner (>=5 sementes), baselines, métricas.
    viz         -- visualizações: superfície de controle, sensibilidade, convergência.

ESTADO (Portão A): este pacote contém apenas STUBS. A lógica dos algoritmos (inferência fuzzy, loop do
PSO, calibração) será implementada no Portão B, após aprovação dos stubs.
"""

__version__ = "0.1.0"
__all__ = ["baseline", "fuzzy", "pso", "reference", "experiments", "viz", "config"]
