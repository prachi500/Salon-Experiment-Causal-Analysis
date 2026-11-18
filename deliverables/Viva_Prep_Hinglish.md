# Viva / interview prep (Hinglish)

**1. A/B test aur DiD mein kya farak hai?**
A/B test mein hum khud randomly log ko do groups mein baantte hain, isliye dono groups barabar hote hain. DiD tab use hota hai jab experiment nahi hua. Wahan hum treated region (Coimbatore) ko control region (Tiruppur, Erode) se compare karte hain, before aur after dono mein, taaki seasonality hat jaye.

**2. Sample ratio mismatch (SRM) kya hai aur kyun check karte hain?**
Agar plan 50/50 tha aur actual split 55/45 aa gaya, to randomisation mein kuch toot gaya hai (bug, tracking issue). Aisa ho to result par bharosa nahi kar sakte. Humare data mein split 49.7% tha, chi-square p = 0.46, matlab koi problem nahi.

**3. Sample size kaise nikala?**
Power analysis se: baseline no-show 12.35%, hum 1.5 percentage point ka farak pakadna chahte the, 80% power, 5% significance. Result 7,147 per arm, 5% loss ke liye 7,524 per arm, total 15,048. Duration: unique clients enrol hone mein 9 hafte lagte hain.

**4. Primary metric no-show rate kyun chuna?**
Reminder seedha isi par asar karta hai, result turant pata chal jata hai, aur events kaafi hain. Retention ke liye 60 din rukna padta hai, isliye wo guardrail hai.

**5. Guardrail kya hota hai?**
Wo metric jo kharab nahi hona chahiye jab hum primary metric sudhaar rahe hain. Jaise opt-outs, revenue per visit, retention.

**6. Parallel trends kya hai? Kaise check kiya?**
Matlab change ke bina Coimbatore aur control region ek jaise chalte. Check: pre-launch trend test (p = 0.89), fake launch (placebo) jisme asar nahi mila (p = 0.91), event study. Ye sab bas assumption ko plausible banate hain, prove nahi karte.

**7. Aapka causal estimate kab galat ho sakta hai?**
(a) Coimbatore mein usi time koi aur event hua (festival, competitor). (b) Seasonality dono regions mein alag hui. (c) Spillover: clients ek region se doosre mein gaye. (d) Sirf 22 outlets aur 12 hafte ka post data, to confidence interval ka lower bound zero ke kareeb hai. (e) Composition badla: walk-ins kam hue, to revenue per visit badh sakta hai bina kisi ke zyada kharch kiye.

**8. CUPED kya hai?**
Pre-experiment behaviour (jaise pehle ka spend) se outcome ka predictable hissa nikaal dena. Variance kam hota hai, to test chhota ho sakta hai. Humare data mein revenue par ~7% variance kam hua (0.7 hafta bacha), no-show par sirf 1% (kyunki haan/na outcome ko past se predict karna mushkil hai).

**9. Data simulated kyun hai?**
Brief mein dataset ki file nahi thi, isliye brief ke hisaab se banaya (fixed seed). Notebook real files par bhi chal jayega.
