import pandas as pd
from dateutil import parser
from pathlib import Path
from utils import (
    normalize, is_filtered, sentiment_label, priority_label, category_label,
    extract_info, generate_reply
)

DATA_PATH = Path("data/Sample_Support_Emails_Dataset.csv")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLASSIFIED_CSV = OUT_DIR / "classified_emails.csv"
STATE_CSV = OUT_DIR / "state.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Normalize
    for col in ["sender", "subject", "body", "sent_date"]:
        df[col] = df[col].apply(normalize)
    # Parse datetime
    df["received_at"] = df["sent_date"].apply(lambda x: parser.parse(x) if x else None)
    return df

def process(df: pd.DataFrame) -> pd.DataFrame:
    # Filter relevant emails
    df = df[df["subject"].apply(is_filtered)].copy()

    df["sentiment"] = df.apply(lambda r: sentiment_label(f"{r['subject']} {r['body']}"), axis=1)
    df["priority"]  = df.apply(lambda r: priority_label(r["subject"], r["body"]), axis=1)
    df["category"]  = df.apply(lambda r: category_label(r["subject"], r["body"]), axis=1)

    # Info extraction
    info = df["body"].apply(extract_info)
    df["phones_found"] = info.apply(lambda x: x["phones_found"])
    df["emails_found"] = info.apply(lambda x: x["emails_found"])
    df["requirements"] = info.apply(lambda x: x["possible_requirements"])

    # Draft reply (templated for MVP)
    df["draft_reply"] = df.apply(
        lambda r: generate_reply(
            r["subject"], r["body"], r["category"], r["sentiment"], r["priority"]
        ),
        axis=1
    )

    # Sort urgent first, newest first
    df = df.sort_values(by=["priority", "received_at"], ascending=[True, False])
    # Map "Urgent" < "Not urgent" for sorting correctness
    priority_order = {"Urgent": 0, "Not urgent": 1}
    df["priority_rank"] = df["priority"].map(priority_order)
    df = df.sort_values(["priority_rank", "received_at"], ascending=[True, False]).drop(columns=["priority_rank"])

    return df

def bootstrap_state(processed_df: pd.DataFrame):
    # Create initial state file if missing
    if not STATE_CSV.exists():
        state = processed_df[["sender","subject"]].copy()
        state["status"] = "Pending"
        state.to_csv(STATE_CSV, index=False)

def main():
    df = load_data()
    print("Rows before filter:", len(df))
    df = df.copy()
    print("Rows after filter:", len(df))
    processed = process(df)
    processed.to_csv(CLASSIFIED_CSV, index=False)
    bootstrap_state(processed)
    print(f"Processed {len(processed)} emails. Saved -> {CLASSIFIED_CSV}")
    print(f"State file -> {STATE_CSV}")

if __name__ == "__main__":
    main()
