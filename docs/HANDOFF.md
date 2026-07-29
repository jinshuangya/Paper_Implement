# プロジェクト引き継ぎメモ（クラウド → ローカル）

最終更新: 2026-07-29

## 1. これは何のプロジェクトか

**Lagrangian Cuts for Two-Stage Stochastic Integer Programs**

論文の再現 + 拡張研究のスキャフォールド:

> Rui Chen and James Luedtke, *On Generating Lagrangian Cuts for Two-Stage
> Stochastic Integer Programs*, arXiv:2106.04023v2 (2022).

論文の核心は、二段階確率整数計画（SIP）の Benders 再定式化に対する
**Lagrangian cut** の生成を高速化すること。係数 `pi` を全錐ではなく
**低次元部分空間（直近の Benders cut 係数の張る空間）** で探索することで
各分離を安価にする、という「restricted separation」がポイント。

ソルバは商用ライセンス不要の **HiGHS (`highspy`)** を使用。全 LP/MIP がこれで動く。

## 2. リポジトリ / ブランチ

- リモート: `jinshuangya/Paper_Implement`（GitHub）
- 作業ブランチ: **`claude/project-local-setup-summary-1hq57h`**
- クラウド側の作業ツリーはクリーン（未コミット変更なし）。最新コミットは
  `81af9bf Add rstrmipV: enrich rstrmip's candidate pool with integer vertices`

ローカルで続けるなら:
```bash
git clone <repo-url> && cd Paper_Implement
git fetch origin claude/project-local-setup-summary-1hq57h
git checkout claude/project-local-setup-summary-1hq57h
```

## 3. コード構成

```
src/lagcuts/
  sslp.py         SSLP インスタンス生成（論文 appendix のレシピ）+ 大M容量ノブ, decoupled 版
  oracle.py       Q_s(x), Benders LP 双対 cut, Q*_s(pi,pi0) 単一シナリオ MIP
  separation.py   Algorithm 1: restricted separation + Pi_s + プール Ehat_s
  rstrmip.py      RstrMIP 基底選択 MIP（eq. 28）
  master.py       Benders マスター（根ノード LP 緩和）, 動的 cut
  algo2.py        Algorithm 2 ループ, 下界 vs 時間ログ, stop_reason 記録
  adaptive.py     ★拡張A: SVD による適応的低次元係数部分空間
  reference.py    検証用 extensive-form z_IP
experiments/
  run_rootnode.py       bound vs time 比較（各手法）
  run_basis_rank.py     拡張Aの診断: Benders スパンが崩壊するか
  run_extensionA_demo.py 拡張A: 収束下界ギャップのデモ
tests/test_validation.py  理論的順序関係の検証
```

実装済みの cut 生成手法（`lagcuts.algo2.Config` の `method=`）:

| method    | Pi_s                                          | 出典 |
|-----------|-----------------------------------------------|------|
| `benders` | Benders cut のみ（LP 緩和の下界）             | §2.1 |
| `exact`   | 正規化した全錐 `alpha*pi0 + ‖pi‖₁ ≤ 1`        | eq.17 |
| `rstr1`   | `pi = Σ βₖ πᵏ`, `alpha*pi0 + ‖pi‖₁ ≤ 1`       | eq.19 |
| `rstr2`   | `pi = Σ βₖ πᵏ`, `alpha*pi0 + ‖β‖₁ ≤ 1`        | eq.20 |
| `rstrmip` | eq.20 で基底を MIP 選択（eq.28）              | §4.3 |
| `rstrmipV`| rstrmip の候補プールを整数頂点で拡張          | 拡張 |
| `adaptiveA`| ★拡張A: SVD 適応部分空間                     | 拡張 |

## 4. 現在の実験状況・効果

### 再現結果（論文の中心的知見を再現できている）

根ノード下界 vs 実時間、SSLP インスタンス `m=10, n=25, S=30, delta=0.5, K=20`:

| method  | final LB | gap closed | oracle(MIP) 呼び出し |
|---------|---------:|-----------:|--------------------:|
| benders | −292.5   |       0 %  |           0 |
| exact   | −153.9   |    97.2 %  |        2019 |
| rstr1   | −159.4   |    93.2 %  |         454 |
| rstr2   | −149.8   |     100 %  |         800 |
| rstrmip | −149.8   |     100 %  |         800 |

- `rstr2`/`rstrmip` が最速で Lagrangian 双対下界に到達。
- `rstr1` は少し低い所で頭打ち（基底が直近 K 個の Benders 係数に限られるため）。
- `exact` は非常に遅く、時間制限内で未収束 → **「restricted separation は
  exact separation より遥かに速く下界を改善する」** という論文の主張を再現。
- 図: `docs/figures/bound_vs_time_sslp1_10_25_30.png`,
  `bound_vs_time_sslp1_8_15_15.png`

### 検証（`tests/test_validation.py`）

- `z_LP ≤ z_restricted ≤ z_LC(exact) ≤ z_IP` の順序関係
- `z_LC = z_D`（Theorem 3）: 小インスタンスで exact separation が
  Lagrangian 双対ギャップを完全に閉じることを確認。

### 拡張A（研究の現状 — ここが「進行中」）

**動機**: rstr1 の頭打ちは、探索方向が Benders LP 接線のスパン内に
限られることが原因。このスパンは条件次第で**ランク落ち**する。

- `run_basis_rank.py`: `capacity_scale`（大M）を掃引すると、LP 緩和が緩む
  につれ **Benders スパンは崩壊するが、Q*_s オラクルが返す整数頂点 x* の
  スパンは満ランクのまま**、という構造信号を確認。図
  `docs/figures/basis_rank_collapse.png`。
- `adaptive.py`: 分離基底を「オンライン維持される生成集合」として扱い、
  候補方向（Benders 係数 **＋** 整数頂点 x*）を L2 正規化して truncated SVD
  し、上位 K 主方向を直交基底として返す。`energy` 指定でスペクトル駆動の
  適応的基底サイズも可能。
- `run_extensionA_demo.py`: decoupled loose-link/tight-capacity インスタンス
  （Benders スパンが崩壊する設計）で、各 restricted 手法を完全収束
  （delta=0）させ、到達下界を比較。Benders 基底の手法が
  `stop_reason == "no_cut"` で双対下界より下で止まる＝天井に当たっている、
  という形で拡張Aの優位性を示す設計。

## 5. 次にやることの提案

**まず（ローカル移行直後の健全性確認）**
1. `pip install -r requirements.txt` 後、`pytest tests -q` が green か確認。
   ※クラウド側では依存が未インストールで最終テスト実行は未完了。
   コードは読み込み済みで論理は健全だが、**ローカルでの実行確認が最初のTODO**。
2. `run_rootnode.py` / `run_extensionA_demo.py` を回して図を再生成し、
   README の数値と一致するか照合。

**拡張Aの本筋（論文化を見据えて）**
3. **理論**: 「cut 強度損失 vs 部分空間近似誤差」の定量バウンドを与える
   （README ロードマップの A で言及済み）。SVD 打ち切り誤差 → cut violation
   の下界劣化、という形で。
4. **実証**: adaptiveA が rstr1/rstr2/rstrmip の頭打ちを実際に超えるかを、
   複数インスタンス・複数 `capacity_scale` で統計的に示す（現状は単発デモ）。
   oracle 呼び出し数あたりの gap-closed で効率も比較。
5. **拡張C（未着手）**: シナリオ横断で良い部分空間/`pi` を予測する学習済み
   共有ポリシーで分離コストを償却。拡張Aの内側に自然に入れ子にできる。

**運用面**
6. 実験の乱数シード・インスタンスパラメータを設定ファイル化し、結果を
   `results/`（gitignore 済み）に構造化保存すると再現性が上がる。
7. CI（GitHub Actions）で `pytest` を回すと、ローカル/クラウド往復での
   デグレを早期検知できる。

## 6. 実行方法（クイックリファレンス）

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests -q            # 検証
PYTHONPATH=src python experiments/run_rootnode.py   # bound vs time
PYTHONPATH=src python experiments/run_basis_rank.py # 拡張A 診断
PYTHONPATH=src python experiments/run_extensionA_demo.py --m 6 --n 12 --S 8 --k 2 --capacity-scale 0.5 --time-limit 300
```
