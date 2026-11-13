from herdmind.brains.registry import make_brains
from herdmind.world.experiment import run_experiment
from herdmind.world.ledger import GRASS, BoardState, read_ledger
from herdmind.world.model import WorldModel
from herdmind.world.spec import WorldSpec


def test_snapshot_matches_the_live_world_and_round_trips():
    w = WorldModel(WorldSpec(), make_brains({"sheep": "rules", "wolf": "rules"}))
    w.step()
    board = w.board()
    assert board.step == 1
    assert {(a.agent_id, a.x, a.y, a.energy) for a in board.animals} == {
        (a.unique_id, a.x, a.y, a.energy) for a in w.animals()
    }
    assert sum(row.count(GRASS) for row in board.grass) == int(w.grass.sum())
    assert BoardState.from_json(board.to_json()) == board


def test_ledger_has_every_step_and_agrees_with_other_logs(tmp_path):
    run = run_experiment(WorldSpec(steps=5), tmp_path)
    ledger = read_ledger(run)
    assert [b.step for b in ledger] == list(range(6))  # step 0 is the starting layout
    assert ledger[0].count("sheep") == 40 and ledger[0].count("wolf") == 8
    replayed = WorldModel(WorldSpec(steps=5), make_brains({"sheep": "rules", "wolf": "rules"}))
    assert ledger[0] == replayed.board()  # same world_seed, same starting board
    for _ in range(5):
        replayed.step()
    assert ledger[-1] == replayed.board()  # same seeds, same board at the end
