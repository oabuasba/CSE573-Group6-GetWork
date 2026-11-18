# Raw data

Two public datasets feed the pipeline. Small files are committed; the large ones are git-ignored
(files over 50 MB, see `.gitignore`) and you place them yourself.

## Job postings: `data/raw/job_postings/`

LinkedIn Job Postings (2023-2024) by arshkon, on Kaggle:
<https://www.kaggle.com/datasets/arshkon/linkedin-job-postings> (citation in
`docs/Group6-Project3-FA26-Group-Project-Presentation.txt`, reference [1]). Download it and
unpack the CSVs so the layout is:

```
job_postings/companies/{companies,company_industries,company_specialities,employee_counts,postings}.csv
job_postings/jobs/{benefits,job_industries,job_skills,salaries}.csv
job_postings/mappings/{industries,skills}.csv
```

Not committed (493 MB): `job_postings/companies/postings.csv`. Place it at exactly that path.
The preprocessor (`run_postings`) reads `companies/postings.csv`, `jobs/job_industries.csv` and
`mappings/industries.csv`.

## IT resumes: `data/raw/resume_corpus/`

Multi-labeled IT resume corpus by Kameni Florentin Flambeau Jiechieu and Norbert Tsopze
(Neural Computing and Applications, 2021, <https://doi.org/10.1007/s00521-020-05302-x>). The
dataset repository is <https://github.com/florex/resume_corpus> (taken from the folder's own
git remote, which was kept in a backup outside this repo). `README.md` in that folder is the
authors' description.

Committed: `README.md`, `normlized_classes.txt`, `skills_it.txt`.

Not committed (over 50 MB), place them in `data/raw/resume_corpus/`:

| File | Size |
| --- | --- |
| `resumes_corpus.zip` (read by `run_resumes`) | 85 MB |
| `resume_samples.zip` | 61 MB |
| `resume_samples.txt` | 204 MB |

Only `resumes_corpus.zip` is needed to rerun preprocessing; `data/processed/` already holds its output.
