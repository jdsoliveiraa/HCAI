# Rating the justifications

`audit_items.csv` has 50 justifications, one per row. Each shows a CV and, for every criterion of the job ad,
the score (0 to 10) a recruiter gave it and the quote from the CV they gave as evidence.

On the project site's audit page (`audit.html`) you label each quote; the item's label is then
`unfaithful` when any quote is. In the spreadsheet, for each row, fill in `label` with one word:

- `faithful`: every quote appears in the CV (verbatim or nearly so) and supports the score it is given;
  an empty quote is fine for a score of 0 when the CV shows nothing for that criterion;
- `unfaithful`: at least one quote is not in the CV, or does not support its score, or a score of 0 has
  an empty quote although the CV does show evidence for the criterion.

`comment` is optional. Rate on your own: do not discuss items with the other rater. Do not change
`item_id`, add or remove rows; save the file as CSV (UTF-8) under a new name, e.g. `labels_<you>.csv`.
