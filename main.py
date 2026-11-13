"""Command-line interface.

  python main.py evaluate                      # compare all models (RMSE, MAE, P/R/NDCG@K)
  python main.py recommend --user 1 --n 10     # top-N for a MovieLens userId
  python main.py similar --title "Toy Story"   # content-based similar movies
Add --synthetic to run offline on generated data.
"""
import argparse

import pandas as pd

from recsys.models import ContentBased
from recsys.evaluation import evaluate_all
from recsys.pipeline import build_models, prepare

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)


def cmd_evaluate(args):
    ds, train, test = prepare(args.synthetic)
    print(f"{ds.n_users} users, {ds.n_items} movies, {len(ds.ratings)} ratings "
          f"(train {len(train)}, test {len(test)})")
    table = evaluate_all(build_models(ds, args.alpha), train, test, ds.n_users, ds.n_items, args.k)
    print("\n", table.sort_values("RMSE").to_string())
    out = "results.csv"
    table.to_csv(out)
    print(f"\nSaved {out}")


def cmd_recommend(args):
    ds, _, _ = prepare(args.synthetic)
    if args.user not in ds.user_ids:
        raise SystemExit(f"userId {args.user} not found (valid range: {ds.user_ids.min()}-{ds.user_ids.max()})")
    u = int(list(ds.user_ids).index(args.user))
    model = build_models(ds, args.alpha)[args.model]
    model.fit(ds.ratings, ds.n_users, ds.n_items)        # final model uses ALL ratings

    hist = ds.ratings[ds.ratings.u == u].sort_values("rating", ascending=False).head(5)
    print(f"\nUser {args.user} top-rated movies:")
    print(ds.movies.loc[hist.i, ["title", "genres"]].assign(rating=hist.rating.values).to_string(index=False))

    items, preds = model.recommend(u, args.n)
    print(f"\nTop {args.n} recommendations ({args.model}):")
    print(ds.movies.loc[items, ["title", "genres"]].assign(predicted_rating=preds.round(2)).to_string(index=False))


def cmd_similar(args):
    ds, _, _ = prepare(args.synthetic)
    hit = ds.movies[ds.movies.title.str.contains(args.title, case=False, regex=False)]
    if hit.empty:
        raise SystemExit(f"No movie matching '{args.title}'")
    idx = int(hit.iloc[0].i)
    items, sims = ContentBased(ds.movies).similar_items(idx, args.n)
    print(f"\nMovies similar to {ds.movies.loc[idx, 'title']}:")
    print(ds.movies.loc[items, ["title", "genres"]].assign(similarity=sims.round(3)).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use generated data (offline)")
    ap.add_argument("--alpha", type=float, default=0.7, help="hybrid weight of collaborative part")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("evaluate"); e.add_argument("--k", type=int, default=10); e.set_defaults(fn=cmd_evaluate)
    r = sub.add_parser("recommend")
    r.add_argument("--user", type=int, required=True)
    r.add_argument("--n", type=int, default=10)
    r.add_argument("--model", default="hybrid",
                   choices=["popularity", "content", "user_cf_cosine", "user_cf_pearson",
                            "item_cf_cosine", "item_cf_pearson", "svd", "hybrid"])
    r.set_defaults(fn=cmd_recommend)
    s = sub.add_parser("similar"); s.add_argument("--title", required=True)
    s.add_argument("--n", type=int, default=10); s.set_defaults(fn=cmd_similar)

    args = ap.parse_args()
    args.fn(args)
