# Pre-registration

Committed on 2026-10-06, before any candidate's marks were downloaded for analysis.

**What had been seen at this point.** The list of 4,750 centres (state, city, centre name, centre code) and the centres' long names. While checking that the data existed, the header, the first rows and the row count (2,333,162) of the marks file. No distribution, centre statistic or city statistic had been computed. The analyst had also read press coverage of the data (Sikar, Rajkot, Hazaribagh, Patna, Godhra), so List B below is not blind.

Any later change to anything in this file is logged, with its date and reason, in `DEVIATIONS.md`. Changes never replace a registered analysis; they are reported next to it.

## The question

In 2024 the Supreme Court ordered the National Testing Agency (NTA) to publish every NEET-UG candidate's marks, centre by centre. It is the only year in which this exists. The court then compared city-level success rates with earlier years and found no sign of a systemic leak.

This project asks three things of that data:

1. **Integrity.** Did any centre produce results that are statistically incompatible with candidates being allotted to centres at random within their city? And what size of fraud could such a screen have caught at all?
2. **Geography.** How concentrated are top scores across India's 571 exam cities, and how much of that tracks district development?
3. **Rank fragility.** How many ranks does one question move, at each part of the score range?

## Data

| Source | Use | Licence |
|---|---|---|
| NTA centre-wise results, released 20 July 2024 (serial number and total marks for every candidate at every centre) | everything | public, court-ordered and anonymised; never redistributed here |
| Mirror of the above by PardhavMaradani on GitHub (CSV) | convenience copy | no licence; used only after verification against NTA's own PDFs |
| NFHS-5 district factsheets (2019-21), compiled by jvargh7/nfhs5_factsheets | district development | MIT |
| Supreme Court, *Vanshika Yadav v. Union of India*, 2024 INSC 568 (2 August 2024) | validation figures | public judgment |

**Verification.** A random sample of 100 centres (seed below), every centre in the validation lists, and every centre the screen flags are re-parsed from NTA's PDFs. The multiset of (serial, marks) pairs must match exactly. Any mismatch stops the pipeline.

**Units.** A candidate is one row. A centre is an NTA centre code. A city is NTA's exam city, identified by state and city name; it is the pool within which NTA allots centres.

**Arithmetic check.** NEET 2024 awarded +4 for a correct answer and -1 for a wrong one, over 180 attempted questions. Every valid total is 4c - w with c + w <= 180. Totals outside that set (for example 717, 718 and 719) are counted and listed, never silently dropped.

## Strand 1: integrity screen (primary)

**Natural experiment.** Candidates choose cities; NTA's software allots the centre within the city. If allotment is random, every centre is a random draw from its city's candidates. That makes each centre testable against the rest of its city, and city-level differences (coaching hubs, development) cancel out by design.

**Primary statistic.** For centre j in city c: k_j, the number of its candidates at or above the national top-1% threshold t1. t1 is the smallest score such that at most 1% of all candidates score t1 or more.

**Layer 1: exact test under random allotment.** Under the null, k_j follows a hypergeometric distribution given the city's candidates and the city's top-1% count. This is Fisher's exact test of the centre against the rest of its city. One-sided p-values (excess) and the lower tail (deficit) are both computed. Cities with one centre are excluded.

**Layer 2: allowance for imperfect randomisation.** Real allotment may sort candidates (for example by paper medium or application order). Sorting inflates every centre's variance by the Kish design effect, 1 + (n_j - 1) rho, where rho is the intra-class correlation of top-1% status among candidates of the same centre, within cities. rho is estimated by the one-way ANOVA estimator on centres nested in cities, after removing the 1% of centres with the largest absolute Layer-1 z-scores (so anomalies cannot inflate the null). Each centre's randomised z-score (below) is divided by the square root of its design effect.

**z-scores.** Discrete p-values are converted to exactly standard-normal z-scores under the null with the randomised probability integral transform, p = P(X > k) + U P(X = k), U uniform, fixed seed.

**Multiplicity.** Benjamini-Hochberg at q = 0.05 across all centres in multi-centre cities. Storey's pi0 is reported.

**Registered flag.** A centre is *flagged* if its Layer-2 one-sided (excess) q-value is at most 0.05. Layer-1 flags are reported alongside but carry no weight if rho is materially above zero.

**Checks on the null (reported whatever they show).**
- Efron's empirical null (central matching) fitted to the z-scores within centre-size bands. Under the sorting model, the null variance grows linearly with centre size, with slope close to rho. If it does not, sorting is not the explanation for any overdispersion, and that is reported.
- Asymmetry: sorting widens both tails; fraud only adds to the upper tail. Using the centre-mean statistic below (which is symmetric under the null), the counts of Layer-2 z-scores above +3 and below -3 are compared.

**Secondary statistics** (computed for every centre, reported, never used for the flag):
- S2: centre mean marks against the city, finite-population z-test, with its own design effect from the ICC of marks.
- S3: top-0.1% count (exact test, as the primary).
- S4: share at or above the national median (exact test).
- S5: Kolmogorov-Smirnov distance between the centre and the rest of its city.

**Two pathways on serial order (exploratory).** Within each centre, NTA lists candidates by serial number. The lag-1 autocorrelation of marks along serial order is computed per centre and pooled.
- If pooled autocorrelation is indistinguishable from zero, serial order carries no seating information and no room-level analysis is attempted.
- If it is clearly positive, serial order reflects something (seating, roll-number order or application order), and a one-dimensional scan statistic over windows of consecutive serials (Kulldorff 1997) looks for room-sized clusters of top scorers, with within-centre permutation p-values and Benjamini-Hochberg across centres. Any such cluster is described as a pattern in serial order, not as a room.

## Strand 1b: detection limits (injection-recovery)

Borrowed from exoplanet surveys, which measure what their pipelines can find by planting synthetic signals in real data. Two fraud models are planted into the real data, one at a time, and the full screen is re-run for the affected city with rho and the Benjamini-Hochberg threshold held at their real-data values.

- **Centre fraud** (an insider at one centre, like the Godhra plan): m candidates at one centre have their marks replaced by draws from the national top-1% marks. m in {5, 10, 20, 40, 80}; centres sampled at random within size bands (<250, 250-499, 500-749, >=750 candidates); 400 injections per cell.
- **City leak** (a paper sold before the exam to candidates whom NTA then allots across the city, like Hazaribagh and Patna): m candidates spread at random over one city's centres, marks replaced the same way. m in {25, 50, 100, 200}.

Reported: detection rate by cell, and the smallest m detected at least 80% of the time.

## Strand 1c: blind validation

`validation/blind_list.csv` was compiled on 2026-10-06 from the Supreme Court judgment, CBI statements and press reports, before marks were analysed.

- **List A** (incidents on record): Oasis School, Hazaribagh (leak origin); Jay Jalaram School, Godhra (foiled insider plan); the Sawai Madhopur wrong-medium centre; the grace-marks centres (Hardayal Public School in Bahadurgarh; Balod). Dantewada and Tura are listed as ambiguous because sources do not name which of the city's centres was affected.
- **List B** (singled out in the press after the 20 July release; not blind): the two R.K. University units in Rajkot.
- **Predictions**, written before seeing results: no List-A centre is flagged for excess. The leak beneficiaries were spread across centres by NTA's allotment and the foiled plan never ran, so a centre-level screen should miss them; that miss is the point of Strand 1b. Grace-marks and wrong-paper centres may show deficits. List B has no directional prediction.

`validation/court_figures.csv` holds the court's city figures. The pipeline must reproduce them: Hazaribagh, Patna and Godhra candidates appearing, and counts within ranks 1,08,000 and 56,000 (577 and 622 marks). Expected agreement: exact, or within 1%, with any gap explained (the 20 July file predates the 26 July revision).

## Strand 2: geography and development (secondary)

- **Concentration.** Shares of top-1% scorers in the ten largest-contributing cities against their shares of candidates; Lorenz curve and Gini across cities.
- **Funnel plot** of city top-1% rates against city size (Spiegelhalter 2005), with control limits widened for between-city overdispersion. Cities above the upper 99.8% limit are *hub cities*.
- **Hub or cluster.** For each hub city, a heterogeneity test across its centres tells apart excess spread across every centre (a city effect, as coaching would produce) from excess in a few centres.
- **Development.** Cities are mapped to NFHS-5 districts by name, with a hand-checked override table; uncertain matches are marked and dropped in a sensitivity run. Development index: first principal component of five district indicators (women 15-49 who are literate; women with 10 or more years of schooling; households with electricity; households using improved sanitation; households using clean cooking fuel), signed so that higher means more developed. Outcomes per district: mean marks, top-1% rate, share at or above the national median. Spearman correlations with 95% bootstrap intervals (2,000 resamples of districts). Prediction: positive correlation between the index and all three outcomes.
- **Positive deviance.** Districts whose top-1% rate sits more than 2 standard deviations above the rate predicted by development, listed, with Kota and Sikar (residential coaching hubs) marked. Ecological correlations describe districts, not candidates (Robinson 1950), and exam city is not home district.

## Strand 3: rank fragility (secondary)

- For every score: candidates tied on it, and ranks crossed by 1, 4 and 5 marks (one question left blank, wrong or right).
- At the court's cut points (577 and 622 marks) and at 450, 550, 650 and 700: ranks per question.
- The disputed physics question: the 20 July file still accepts two options. The 26 July revision took 5 marks from candidates who chose the withdrawn option. Bounds on how far that moved ranks are computed from the distribution alone.
- An illustrative measurement-error band from Lord's binomial error model, with every assumption stated, translated into ranks.

## Null results

- If no centre is flagged at Layer 2: the report says that no centre is anomalous beyond allotment noise, and states the smallest frauds the screen would have caught. That is a finding, not a failure.
- If centres are flagged: each is re-verified against NTA's PDF, shown against its city, and described as statistically anomalous under random allotment. Flags are screening results, never findings of misconduct. Known innocent explanations (candidates from one coaching institute allotted together, medium-based allotment, data errors) are discussed for each.

## Fixed choices

| Choice | Value |
|---|---|
| Seed for every random draw | 20261006 |
| Top threshold (primary) | national top 1% |
| False discovery rate | q = 0.05, Benjamini-Hochberg |
| Trim before estimating rho | 1% of centres by absolute Layer-1 z |
| Bootstrap resamples | 2,000 |
| Permutations (serial-order tests) | 10,000 |
| Injections per cell | 400 |
| Verification sample | 100 random centres plus all listed and flagged centres |

## Out-of-sample pathway

NEET-UG 2026 was cancelled on 12 May 2026 after a leak and re-held on 21 June 2026. If NTA has published centre-wise results for the re-exam, the frozen pipeline is run on them unchanged, as a confirmatory test. If not, the report says so.

## Ethics

The data are anonymised and published by order of the Supreme Court. No attempt is made to identify any candidate. Centre names appear only because NTA published them. The report's language is fixed in advance: "anomalous" and "inconsistent with random allotment", never "cheated".
