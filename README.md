# Problem Set 2: The Underpowered Portfolio

**Objective.** I checked whether the client's "ship if p < 0.05" rule picks real winners, using 200 simulated A/B tests where I know the true effect of each test.

**Methodology.** I simulated 200 conversion tests. 80% of ideas have no effect and 20% have a real effect. Sample sizes go from 2,000 to 100,000 users per arm, so some tests have low power and some have high power. I applied the client's rule, measured the winner's curse by power group, bootstrapped the realised return, compared the client's rule with a power-floor rule, and checked an AI-written sequential test on 1,000 null streams.

**Key Findings.** Only 10 of the 22 shipped tests were real wins. Shipped estimates were about 3.4 times too big, and about 12 times too big in low-power tests. The power floor cuts false launches from 54.5% to 30.0%, but gives less realised lift. Gemini's sequential test had a 25.8% false positive rate on my data. After my fix it was 5.6%.
