# SAWS Data Request — WCS-CLS-FRM-004.1 Fill-In

**Generated:** 2026-08-03 · **Corrected:** 2026-08-03 (dropped Bloemfontein WO + Knellpoortdam)
**Purpose:** Pre-2016 backfill for 12 weather stations to expand the existing 26-station 2013–2020 network to a 38-station 2012–2020 network, still under 10% missingness and following Davidson (2022) methodology.

> **⚠ Scope note (2026-08-03):** An earlier version of this request listed 14 stations. Two of them — BLOEMFONTEIN WO and KNELLPOORTDAM — were included on the false premise that their 2012–2015 records in Lekuba data2.xls covered only Temperature and Rain. **In fact data2.xls covers all 6 features for those years**, so no backfill is needed for those two stations. They have been removed from this request. See [AUDIT.md](AUDIT.md) for the corrected data2 analysis.
>
> The 26-station 2013–2020 network is already achievable from data in hand without this request. This request is for the optional expansion to 38 stations (adding 12 Free State / Eastern Cape / KZN interior / Gauteng edge stations that only exist in Gen's block-format extract, which starts 2016).

This document is the paste-ready content for [WCS-CLS-FRM-004.1.doc](../../WCS-CLS-FRM-004.1.doc). The .doc file itself must be edited in Microsoft Word / LibreOffice; this file mirrors it section-by-section for copy-paste.

---

## SCHEDULE 1 — FULL PERSONAL DETAILS OF USER

| Field | Value |
|---|---|
| Full Names | **[FILL IN — Praise Jaravani]** |
| University/school/organisation | **University of Cape Town** |
| Student Number (if applicable) | **[FILL IN]** |
| Email address | **jaravanipraise@gmail.com** |
| Current registered degree | **BSc Honours in Computer Science (CSC4019Z)** |
| Expected finalization date (MMYYYY) | **[FILL IN — likely 122026 if graduating end-2026]** |

---

## THE PURPOSE

*(Paste this whole block into the "THE PURPOSE" field.)*

> This request is for weather station data required for an honours research project (WeatherFLOW) at the University of Cape Town under the supervision of Prof. Deshendran Moodley (Centre for Artificial Intelligence Research). The project is a benchmark study of spatial-temporal graph neural network (ST-GNN) architectures for hourly weather forecasting over South Africa, comparing recent architectures (AGCRN, CLCRN, MTGODE) against baselines, with prediction horizons of 24 hours and beyond.
>
> The study directly extends the methodology of Davidson & Moodley (2022), "ST-GNNs for Weather Prediction in South Africa" (SACAIR 2022, doi:10.1007/978-3-031-22321-1_7), which used 21 SAWS weather stations from the Western and Northern Cape provinces over 2012–2019 with the six standard surface observations (temperature, rain, humidity, wind speed, wind direction, pressure) at hourly resolution. Our study extends Davidson's network geographically to include Eastern Cape, Free State, and KwaZulu-Natal stations so we can evaluate whether ST-GNN spatial dependency learning generalises across South Africa's climatically heterogeneous regions (winter-rainfall Cape vs summer-rainfall east).
>
> We already possess SAWS data for the 12 stations in this request covering 2016 onwards (received in a prior extract). We are requesting the earlier 2012–2015 records for the same stations, so that all stations in the study cover the same uniform 2012–2020 period matching Davidson's window. This is required for the methodological consistency of the walk-forward validation framework used in Davidson (2022).
>
> The research is non-commercial, produced solely as an honours-thesis output; no financial benefit is anticipated from the research findings. SAWS will be acknowledged in the resulting thesis and any subsequent publication, and a copy of the final research output will be provided to SAWS on completion.

---

## DATA REQUIRED

*(Paste this whole block into the "DATA REQUIRED" field.)*

> **Weather elements (all 6):**  Temperature, Rainfall, Relative Humidity, Wind Speed, Wind Direction, Barometric Pressure.
>
> **Resolution:**  Hourly surface observations.
>
> **Time period:**  01 January 2012 to 31 December 2015 (inclusive) — 4 years.
>
> **Stations (12 total, sorted by climate number):**
>
> | # | Station Name | Climate Number | Latitude | Longitude | Elevation (m) |
> |---|---|---|---|---|---|
> | 1 | ADDO ELEPHANT PARK | 0055447 | −33.4420 | 25.7480 | 142 |
> | 2 | SOMERSET EAST | 0055363 1 | −33.0490 | 25.7190 | 513 |
> | 3 | GRAAFF-REINET | 0096072 5 | −32.1930 | 24.5430 | 792 |
> | 4 | ELLIOT | 0150620 | −31.3360 | 27.8400 | 1464 |
> | 5 | ALIWAL-NORTH PLAATKOP | 0175678 | −30.8020 | 26.8830 | 1350 |
> | 6 | MARGATE | 0182591 | −30.8580 | 30.3430 | 152 |
> | 7 | PIETERMARITZBURG | 0239698 5 | −29.6270 | 30.4020 | 673 |
> | 8 | MOUNT EDGECOMBE | 0241072 9 | −29.7060 | 31.0460 | 94 |
> | 9 | FAURESMITH | 0291570 1 | −29.7530 | 25.3230 | 1387 |
> | 10 | BETHLEHEM WO | 0331585 9 | −28.2490 | 28.3340 | 1689 |
> | 11 | RIVERVIEW | 0339327 3 | −28.4440 | 32.1820 | 49 |
> | 12 | PRINS ALBERT — SWARTRIVIER | [SAWS to confirm] | ≈ −33.23 | ≈ 22.02 | [SAWS to confirm] |
>
> Notes on individual stations:
> - Stations #1–11: we have 2016–2025 records; requesting 2012–2015.
> - Station #12 (PRINS ALBERT — SWARTRIVIER): we have partial 2013–2020 records via a prior extract; requesting the complete calendar year 2012 to close the coverage gap. If the Prins Albert – Swartrivier station has a different SAWS name we would appreciate SAWS's canonical identifier being included in the reply.
>
> If the earliest available records for any of the listed stations are later than 01 January 2012, whatever period is available would still be very valuable to us.

---

## SIGNATURE / POPIA ACCEPTANCE

- Signature of the User: **[SIGN]**
- Date: **[FILL — e.g. 04 August 2026]**
- POPIA acknowledgement: **[TICK / SIGN]** — I have read and understood the customer processing POPIA notice at https://www.weathersa.co.za/home/popia

---

## Provenance of this request (for your own records)

The 12 stations were identified via the audit at [AUDIT.md](AUDIT.md). Selection criterion:

- **Base network (in hand, no request needed):** 26 stations at 10% missingness on all 6 features over 2013–2020: 17 from Lekuba data1.xlsx (W/N Cape + parts of E Cape) + 9 from Lekuba data2.xls (Free State + KZN + Gauteng edge, all except Ladysmith).
- **Expansion (this request):** 12 additional stations that appear in Gen's block-format extract for 2016–2025 but have no pre-2016 records anywhere in our current data. Their 2016-onwards missingness is 1.3%–7.7% (all well within the 10% Davidson threshold), so we expect a similar-quality pre-2016 extract from SAWS would slot in cleanly.

If the request is fulfilled in full, the resulting corpus is:

- **38 stations × 6 features × 9 years (2012–2020) × hourly**, all under 10% missingness before IDW imputation
- Five provinces (Western Cape, Northern Cape, Eastern Cape, Free State, KwaZulu-Natal)
- Direct methodological parity with Davidson (2022), which used 21 stations × 6 features × 8 years (2012–2019) × hourly in Western + Northern Cape only.

If the request cannot be fulfilled, the base 26-station 2013–2020 network is still viable and remains the fallback plan.
