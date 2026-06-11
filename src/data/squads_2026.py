"""
Goalscorer priors for FIFA World Cup 2026 teams.

For each national team we store its likely scorers and a relative *scoring share*
(roughly, the fraction of the team's goals that player is expected to contribute).
Shares per team sum to < 1.0 — the residual is implicitly spread over the rest of
the squad (midfielders, defenders, own goals) and surfaced as "Other".

Data reflects 2025-26 roles (main striker / penalty taker / top scorer).  Teams
without an entry fall back to a generic "squad" distribution in the goals model.
These are approximate priors for a portfolio prediction tool, not betting advice.
"""
from __future__ import annotations

SCORER_SHARES = {
    "Argentina": [("Lautaro Martinez", 0.26), ("Julian Alvarez", 0.22), ("Lionel Messi", 0.20), ("Nicolas Gonzalez", 0.08), ("Giuliano Simeone", 0.06)],
    "Brazil": [("Vinicius Junior", 0.24), ("Raphinha", 0.16), ("Rodrygo", 0.15), ("Endrick", 0.14), ("Matheus Cunha", 0.10)],
    "France": [("Kylian Mbappe", 0.34), ("Ousmane Dembele", 0.16), ("Marcus Thuram", 0.13), ("Bradley Barcola", 0.10), ("Michael Olise", 0.10)],
    "England": [("Harry Kane", 0.34), ("Bukayo Saka", 0.15), ("Cole Palmer", 0.13), ("Phil Foden", 0.10), ("Anthony Gordon", 0.07)],
    "Spain": [("Lamine Yamal", 0.20), ("Mikel Oyarzabal", 0.18), ("Alvaro Morata", 0.15), ("Nico Williams", 0.13), ("Dani Olmo", 0.12)],
    "Portugal": [("Cristiano Ronaldo", 0.28), ("Bruno Fernandes", 0.16), ("Goncalo Ramos", 0.14), ("Rafael Leao", 0.12), ("Pedro Neto", 0.08)],
    "Germany": [("Florian Wirtz", 0.17), ("Kai Havertz", 0.16), ("Jamal Musiala", 0.16), ("Niclas Fullkrug", 0.14), ("Serge Gnabry", 0.10)],
    "Netherlands": [("Memphis Depay", 0.24), ("Cody Gakpo", 0.18), ("Xavi Simons", 0.13), ("Donyell Malen", 0.12), ("Wout Weghorst", 0.10)],
    "Belgium": [("Romelu Lukaku", 0.28), ("Kevin De Bruyne", 0.16), ("Jeremy Doku", 0.12), ("Leandro Trossard", 0.10), ("Charles De Ketelaere", 0.09)],
    "USA": [("Christian Pulisic", 0.30), ("Folarin Balogun", 0.20), ("Ricardo Pepi", 0.14), ("Tim Weah", 0.10), ("Gio Reyna", 0.08)],
    "Mexico": [("Raul Jimenez", 0.24), ("Santiago Gimenez", 0.22), ("Hirving Lozano", 0.14), ("Alexis Vega", 0.10), ("Cesar Huerta", 0.08)],
    "Canada": [("Jonathan David", 0.30), ("Cyle Larin", 0.16), ("Alphonso Davies", 0.16), ("Tani Oluwaseyi", 0.09), ("Jonathan Osorio", 0.07)],
    "Croatia": [("Andrej Kramaric", 0.20), ("Bruno Petkovic", 0.16), ("Ante Budimir", 0.16), ("Luka Modric", 0.12), ("Igor Matanovic", 0.09)],
    "Uruguay": [("Darwin Nunez", 0.26), ("Federico Valverde", 0.16), ("Facundo Pellistri", 0.10), ("Maximiliano Araujo", 0.10), ("Rodrigo Aguirre", 0.10)],
    "Colombia": [("Luis Diaz", 0.26), ("James Rodriguez", 0.16), ("Jhon Duran", 0.14), ("Rafael Santos Borre", 0.10), ("Luis Suarez", 0.10)],
    "Ecuador": [("Enner Valencia", 0.26), ("Kevin Rodriguez", 0.14), ("Gonzalo Plata", 0.14), ("Kendry Paez", 0.10), ("Jeremy Sarmiento", 0.08)],
    "Paraguay": [("Antonio Sanabria", 0.22), ("Julio Enciso", 0.16), ("Miguel Almiron", 0.14), ("Diego Gomez", 0.10), ("Ramon Sosa", 0.10)],
    "Morocco": [("Youssef En-Nesyri", 0.24), ("Brahim Diaz", 0.14), ("Hakim Ziyech", 0.12), ("Ayoub El Kaabi", 0.10), ("Achraf Hakimi", 0.10)],
    "Senegal": [("Sadio Mane", 0.24), ("Nicolas Jackson", 0.18), ("Ismaila Sarr", 0.14), ("Habib Diallo", 0.10), ("Iliman Ndiaye", 0.09)],
    "Japan": [("Takefusa Kubo", 0.18), ("Kaoru Mitoma", 0.16), ("Ayase Ueda", 0.16), ("Daizen Maeda", 0.12), ("Junya Ito", 0.10)],
    "South Korea": [("Son Heung-min", 0.30), ("Hwang Hee-chan", 0.16), ("Lee Kang-in", 0.14), ("Oh Hyeon-gyu", 0.10), ("Cho Gue-sung", 0.09)],
    "Australia": [("Mitchell Duke", 0.18), ("Jackson Irvine", 0.14), ("Martin Boyle", 0.14), ("Kusini Yengi", 0.12), ("Craig Goodwin", 0.10)],
    "Iran": [("Mehdi Taremi", 0.30), ("Sardar Azmoun", 0.22), ("Alireza Jahanbakhsh", 0.12), ("Mehdi Ghayedi", 0.09), ("Shahriar Moghanlou", 0.07)],
    "Saudi Arabia": [("Salem Al-Dawsari", 0.24), ("Firas Al-Buraikan", 0.20), ("Saleh Al-Shehri", 0.14), ("Abdullah Al-Hamdan", 0.10), ("Musab Al-Juwayr", 0.08)],
    "Switzerland": [("Breel Embolo", 0.22), ("Dan Ndoye", 0.14), ("Ruben Vargas", 0.14), ("Zeki Amdouni", 0.12), ("Granit Xhaka", 0.10)],
    "Ghana": [("Mohammed Kudus", 0.22), ("Inaki Williams", 0.16), ("Jordan Ayew", 0.16), ("Antoine Semenyo", 0.12), ("Ernest Nuamah", 0.08)],
    "Ivory Coast": [("Sebastien Haller", 0.20), ("Simon Adingra", 0.16), ("Nicolas Pepe", 0.14), ("Franck Kessie", 0.12), ("Jeremie Boga", 0.09)],
    "Norway": [("Erling Haaland", 0.42), ("Alexander Sorloth", 0.18), ("Martin Odegaard", 0.12), ("Antonio Nusa", 0.08), ("Oscar Bobb", 0.06)],
    "Egypt": [("Mohamed Salah", 0.36), ("Omar Marmoush", 0.18), ("Mostafa Mohamed", 0.14), ("Trezeguet", 0.09), ("Ahmed Sayed Zizo", 0.07)],
    "Tunisia": [("Hannibal Mejbri", 0.16), ("Youssef Msakni", 0.16), ("Naim Sliti", 0.14), ("Elias Achouri", 0.12), ("Seifeddine Jaziri", 0.10)],
    "Algeria": [("Mohamed Amoura", 0.22), ("Riyad Mahrez", 0.20), ("Baghdad Bounedjah", 0.14), ("Said Benrahma", 0.10), ("Amine Gouiri", 0.10)],
    "Austria": [("Marko Arnautovic", 0.20), ("Michael Gregoritsch", 0.16), ("Marcel Sabitzer", 0.14), ("Christoph Baumgartner", 0.12), ("Patrick Wimmer", 0.08)],
    "Turkey": [("Arda Guler", 0.20), ("Kerem Akturkoglu", 0.16), ("Hakan Calhanoglu", 0.16), ("Baris Alper Yilmaz", 0.12), ("Kenan Yildiz", 0.12)],
    "Sweden": [("Alexander Isak", 0.30), ("Viktor Gyokeres", 0.26), ("Anthony Elanga", 0.12), ("Dejan Kulusevski", 0.10), ("Emil Forsberg", 0.08)],
    "Scotland": [("Che Adams", 0.20), ("Scott McTominay", 0.18), ("John McGinn", 0.14), ("Lyndon Dykes", 0.12), ("Lawrence Shankland", 0.10)],
    "Czechia": [("Patrik Schick", 0.30), ("Adam Hlozek", 0.16), ("Tomas Soucek", 0.14), ("Antonin Barak", 0.10), ("Mojmir Chytil", 0.08)],
    "Bosnia and Herzegovina": [("Edin Dzeko", 0.30), ("Ermedin Demirovic", 0.20), ("Smail Prevljak", 0.12), ("Said Hamulic", 0.10), ("Benjamin Tahirovic", 0.06)],
    "South Africa": [("Percy Tau", 0.22), ("Lyle Foster", 0.20), ("Themba Zwane", 0.12), ("Evidence Makgopa", 0.10), ("Relebohile Mofokeng", 0.08)],
    "Qatar": [("Almoez Ali", 0.28), ("Akram Afif", 0.22), ("Hassan Al-Haydos", 0.12), ("Yusuf Abdurisag", 0.10), ("Ahmed Alaaeldin", 0.06)],
    "Haiti": [("Frantzdy Pierrot", 0.26), ("Duckens Nazon", 0.18), ("Danley Jean Jacques", 0.12), ("Ruben Providence", 0.10), ("Don Deedson Louicius", 0.08)],
    "Cape Verde": [("Garry Rodrigues", 0.22), ("Ryan Mendes", 0.16), ("Bebe", 0.14), ("Jovane Cabral", 0.12), ("Kevin Pina", 0.08)],
    "New Zealand": [("Chris Wood", 0.34), ("Ben Old", 0.14), ("Kosta Barbarouses", 0.12), ("Eli Just", 0.10), ("Matthew Garbett", 0.08)],
    "Iraq": [("Aymen Hussein", 0.26), ("Mohanad Ali", 0.20), ("Ali Al-Hamadi", 0.16), ("Amir Al-Ammari", 0.10), ("Ibrahim Bayesh", 0.07)],
    "Jordan": [("Mousa Al-Taamari", 0.28), ("Yazan Al-Naimat", 0.20), ("Ali Olwan", 0.14), ("Mahmoud Al-Mardi", 0.10), ("Nizar Al-Rashdan", 0.07)],
    "DR Congo": [("Cedric Bakambu", 0.22), ("Yoane Wissa", 0.20), ("Fiston Mayele", 0.14), ("Silas Katompa", 0.12), ("Theo Bongonda", 0.10)],
    "Uzbekistan": [("Eldor Shomurodov", 0.30), ("Abbosbek Fayzullaev", 0.16), ("Igor Sergeev", 0.14), ("Jaloliddin Masharipov", 0.10), ("Ostonali Umarov", 0.07)],
    "Panama": [("Cecilio Waterman", 0.22), ("Jose Fajardo", 0.18), ("Ismael Diaz", 0.16), ("Eric Davis", 0.10), ("Adalberto Carrasquilla", 0.08)],
    "Curacao": [("Tahith Chong", 0.22), ("Gervane Kastaneer", 0.16), ("Juninho Bacuna", 0.14), ("Kenji Gorre", 0.12), ("Leandro Bacuna", 0.10)],
}

# Generic distribution for teams without a specific entry: one focal forward plus
# a couple of secondary contributors, leaving a large residual for the rest.
GENERIC_SHARES = [
    ("Main striker", 0.22),
    ("Secondary forward", 0.14),
    ("Attacking midfielder", 0.10),
]


def get_scorer_shares(team: str):
    """Return [(player, share), ...] for a team, or a generic profile if unknown."""
    return SCORER_SHARES.get(team, GENERIC_SHARES)
