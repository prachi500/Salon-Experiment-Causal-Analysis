# Data dictionary

## appointments.csv.gz (271,850 rows, including ~1,000 duplicate rows)
| Column | Meaning |
|---|---|
| appt_id | Appointment ID |
| appt_date, appt_hour | Date and start hour |
| outlet_id | Outlet (O01 to O22) |
| client_id | Client ID; **empty for walk-ins** |
| booking_type | Booked or Walk-in |
| service_raw | Service name as typed at the outlet (37 spellings) |
| stylist_id | Stylist |
| status | Completed or Cancelled. **Cancelled also contains no-shows** |
| cancel_lead_hours | Hours before the slot the cancellation was made. Empty or <= 0 means no cancellation before the slot, i.e. a no-show |
| service_price, addon_price | Price of the service and premium add-on (only for completed visits) |

## product_sales.csv.gz
sale_id, sale_date, outlet_id, appt_id (empty for standalone retail), product_category, amount.

## outlets.csv
outlet_id, city (Coimbatore, Tiruppur, Erode), area, format (Salon, Salon & Spa, Spa), change_launched (Y = Coimbatore, launched 1 Jul 2026), seats, open_year.

## stylists.csv
stylist_id, outlet_id, level (Junior, Senior, Master), tenure_months.

## experiment_clients.csv (14,000 clients, WhatsApp reminder pilot, enrolled 2 Mar to 26 Apr 2026)
client_id, outlet_id, enrol_date, index_appt_date, arm (control or treatment), is_new_client, age_group, gender, booking_channel, service_category, weekend_appt, pre_visits_12m, pre_avg_spend_inr, pre_noshow_rate (pre-experiment covariates; empty for new clients), status, no_show, completed, revenue_inr, opted_out, retained_60d (outcomes).
