# Dialect test, side by side

What the models write for the same audio: segments from the SKN dialect test ([dataset](https://huggingface.co/datasets/GAIK-project/skn-finnish-dialects-asr),
split `test`), 10 municipalities that are not in the training data. **Human** is SKN's transcript in
standard Finnish word forms; **Whisper** is plain openai/whisper-large-v3; **Ours** is the
[published model](https://huggingface.co/GAIK-project/whisper-large-v3-fi-dialects).
The numbers are word errors / reference words after lower-casing and removing punctuation.

The examples are picked by position, not by result: per municipality, of the segments with 12-40
reference words in id order, the ones one third and two thirds of the way through. Find a segment's
audio and dialect-spelling transcript in the dataset viewer by its id. Written by
[`dialect_examples.py`](../scripts/dialect_examples.py).

| Municipality | Segments | Whisper large-v3 WER % | Ours WER % |
|---|--:|--:|--:|
| Alastaro | 60 | 35.7 | 17.6 |
| Artjarvi | 60 | 44.4 | 23.1 |
| Eurajoki | 60 | 43.7 | 22.0 |
| Heinola | 60 | 36.1 | 19.9 |
| Hinnerjoki | 60 | 51.8 | 29.0 |
| Ikaalinen | 60 | 31.3 | 15.0 |
| Jaala | 60 | 37.3 | 18.2 |
| Mikkeli | 60 | 38.5 | 19.9 |
| Rautu | 60 | 50.4 | 31.6 |
| Vihti | 60 | 32.8 | 14.4 |
| **All** | 600 | **40.4** | **21.1** |

## Examples

**Alastaro** · `SKN25a_Alastaro_0156`

| | Text | Errors |
|---|---|--:|
| Human | niin hahtuvakoppa pantiin sitten kanan päälle alassuin kun se sängyn jalkopäähän otettiin munille | |
| Whisper | Niin, haastava koppapa. Niin sit kanan päällä alas syiden, kun se sänky jalkapäähän nyt etsii munille. | 11 / 13 |
| Ours | Niin, hastalokoppa pantiin sitten kanan päälle alas suiten, kun se sänky jälkipäähän nytettiin munille. | 6 / 13 |

**Alastaro** · `SKN25b_Alastaro_0086`

| | Text | Errors |
|---|---|--:|
| Human | ja siellä oli - naiset oli tavallansa olivat riihessä niitä lyömässä seinään sitten ne annettiin siitä toiseen huoneeseen siellä oli sitten miehiä ja voi joskus naisiakin neljä on tavallansa | |
| Whisper | Ja siellä oli, naiset oli tavallaan, se oli riihessä niitä lyömässä seinään. Sitten ne annettiin sitten toiseen huoneeseen. Siellä oli sitten miehiä ja joskus naisiakin. Neljä tavallaan se. | 7 / 28 |
| Ours | Ja siellä oli, naiset oli tavallaan, se oli riihessä niitä räämässä seinään, sitten ne annettiin sitten toiseen huoneeseen, siellä oli sitten miehiä ja joskus naisiakin neljä tavallaan. | 8 / 28 |

**Artjarvi** · `SKN35a_Artjarvi_0132`

| | Text | Errors |
|---|---|--:|
| Human | puhuiko ne mitään semmoisesta taudista jota sanottiin mustaksi surmaksi niin en minä ole sellaista kuullut niin kyllä minä olen kuullut puhuttavan niin mutta en minä ainakaan nyt täälläpäin ole kuullut sellaista tautia | |
| Whisper | Puhuuko ne mitään sellaista taulusta, jota sanottiin mustaksi surmaksi? Niin, en minä ole sellaista kuullut. Kyllä minä olen kuullut puhuttavani, mutta en minä ainakaan nyt täälläpäin ole kuullut sellaista autia. | 7 / 32 |
| Ours | Puuko ne mitä semmoista taulusta, jota sanottiin mustaksi surmaksi? Niin en minä ole sellaista kuullut, niin kyllä minä olen kuullut puhuttavani, mutta en minä ainakaan nyt täällä päin ole kuullut sellaista tautia. | 8 / 32 |

**Artjarvi** · `SKN35b_Artjarvi_0094`

| | Text | Errors |
|---|---|--:|
| Human | mutta että on niitä nyt vieläkin vanhempia yksilöitä että on - elänyt vielä minunkin muistoon vaikka ei meillä ole ollut mutta nuo - noilla muilla | |
| Whisper | Mutta että on niitä nyt vieläkin vanhempia yksilöitä, että on elänyt vielä minkin muistoon, vaikka ei meillä ole ollut, mutta noin muilla. | 3 / 23 |
| Ours | Mutta että on niitä nyt vieläkin vanhempia yksilöitä, että on elänyt vielä minkin muista, vaikka ei meillä ole ollut, mutta noin, noin muilla. | 4 / 23 |

**Eurajoki** · `SKN50a_Eurajoki_0132`

| | Text | Errors |
|---|---|--:|
| Human | ja sen saa siitä sitten aina että mitä kauemmin vain keittää niin sitä sakeampaa tulee mutta jos hiukan keittää niin se oli semmoista löysää | |
| Whisper | Ja sen saa siitä sitten aina, että mitä kauemmin vaan keittää, niin sitä sakkempaa tulee, mutta jos hiukan keittää, niin se on semmoista löysää. | 3 / 24 |
| Ours | Ja sen saa siitä sitten aina, että mitä kauemmin vain keittää, niin sitä sakempaa tulee, mutta jos hiukan keittää, niin se on semmoista löysää. | 2 / 24 |

**Eurajoki** · `SKN50a_Eurajoki_0203`

| | Text | Errors |
|---|---|--:|
| Human | ja sitten kun niitä niitä pentuja penikoita tuli niin ne pantiin sitten yhden kanan perään kaikki yksi kana hoiti ne sitten saatiin ne toiset kaksi kanaa jälleen munimaan | |
| Whisper | Ja sit kun niitä penikoi tuli, niin ne panti sit yhden kanan perään kaikki. Yks kana hoitnes sitten. Saati ne toiset kaks kanaa ja lemuniimaa. | 13 / 28 |
| Ours | Ja sitten kun niitä pentuja, penikoita tuli, niin ne pantiin sitten yhden kanan perään kaikki, että yksi kana hoiti ne sitten, saatiin ne toiset kaksi kanaa jälleen monimaan. | 3 / 28 |

**Heinola** · `SKN45a_Heinola_0112`

| | Text | Errors |
|---|---|--:|
| Human | kun ne tulee hampaat pois niin niille tulee se syöminen huonoksi niin - mitäs se ja sitten ei se huontuu niin että sitä ei enää hirveä eikä viitsi pitääkään mm | |
| Whisper | Kun ne tuli hampaat pois, niin ne tuli, ja se syömii huonoks nimisessä, ja sitten ei se huontu, niin että sitä ei enää hirveä aikaa viitti pitikin. | 14 / 29 |
| Ours | Kun ne tulee hampaat pois, niin ne tulee se syömiä huonoksi, niin missä, ja sitten ei se huontuu niin, että sitä ei enää hirveänä kävitse pitikään, mm. | 8 / 29 |

**Heinola** · `SKN45b_Heinola_0044`

| | Text | Errors |
|---|---|--:|
| Human | mutta kun ne niitä kaskia poltti niin sitten ne oli taas erilaisia | |
| Whisper | Mut kun ne niit kaskia poltti, niin sitten ne oli taas erilaisia. | 2 / 12 |
| Ours | Mutta kun ne niitä kaskia poltti, niin sitten ne oli taas erilaisia. | 0 / 12 |

**Hinnerjoki** · `SKN30a_Hinnerjoki_0095`

| | Text | Errors |
|---|---|--:|
| Human | aika valo oli siellä patsaalla sitten tuota ja - että lämmin oli ja yksi piti hevosen jalkaa sitten ylhäällä ja toinen sovitteli sitten kenkää siihen ja oli hohtimet ja vasara ja puukko sen kädessä sitten siinä tykönään | |
| Whisper | Aika valuus ja passas sit tätä ja lämmin oli ja yks pit hevosen jalkka sit ylhäällä ja toinen sovitti ja sit kenkkä siihen ja oli hohtime ja vasara ja puukko sen kädessä sit ylhäällä. | 19 / 36 |
| Ours | Aika valossa ja passassa sitten tuota, ja että että lämmin oli, ja yksi piti hevosen jalka sitten ylhäälle, ja toinen sovitti ja sitten kenkkä siihen, ja oli hohtimen ja vasaran ja puukko sen käden sitten ylös. | 15 / 36 |

**Hinnerjoki** · `SKN30b_Hinnerjoki_0060`

| | Text | Errors |
|---|---|--:|
| Human | kyllä se on sitten jo vilheän näköinen kun porsas on kiskottu en minä ole ikinä nähnyt semmoista kiskottua sikaa | |
| Whisper | Kyllä se on sitten jo vilhijänäköinen, kun porsas on iskottu. Ja minä olinkin aina semmoista iskottu rikkaalla. | 9 / 19 |
| Ours | Kyllä se on sitten jo vilheä näköinen, kun korsassa on iskot. En minä ole ikänä semmoista iskottu sittenkaan. | 7 / 19 |

**Ikaalinen** · `SKN20b_Ikaalinen_0050`

| | Text | Errors |
|---|---|--:|
| Human | jos hän sitten pitää paikkansa no minkälainen uni se oli oli niin mahdottoman suuri tiinu semmoinen oikein suuri tiinu niin kuin lehmien haudepunttu ja se oli täynnä niin sotkettuna sitten niin ruisleipätaikinaa | |
| Whisper | jos hän sitä pitää paikkaansa. Minkälainen uni se oli? Oli niin mahdottoman suuri tiinu, semmoinen oikein suuri tiinu, niin kuin lehmä ja haulepunttu. Se oli täynnä niitä sotkettua ruihleipataakin. | 11 / 32 |
| Ours | Jos hän sitten pitää paikkaansa. No minkälainen uni se oli? Oli niin mahdottoman suuri tiinu, semmoinen oikein suuri tiinu, niin kuin lehmän ja haudat punttu, ja se oli täynnä niin sotkettua sitten, niin ei ne ruisleipätaakin ole. | 10 / 32 |

**Ikaalinen** · `SKN20b_Ikaalinen_0143`

| | Text | Errors |
|---|---|--:|
| Human | no millä lailla se semmoinen tonttu kulkee - hyppii sillä jalallansa niin | |
| Whisper | No millä lailla se semmoinen tonttu kulkee? Hyppii. Isillä ja laulamisilla. | 3 / 11 |
| Ours | No millä lailla se semmoinen tonttu kulkee? Hyppiä sillä jalallaan, niin. | 2 / 11 |

**Jaala** · `SKN40a_Jaala_0091`

| | Text | Errors |
|---|---|--:|
| Human | sitten tehtiin sellaisia - tuon - niin kuin kynttilä on tuon pituisia lankoja ja sitten pantiin se tikku siihen nenään niin kun nyt kynttilässä on se nenä mistä mihinkä tuli laitetaan | |
| Whisper | Sitten tehtiin sellaisia pitkää, niin kuin kynttilä on, tompituslankoja. Sitten pantiin se tikku siihen nenään, niin kuin nyt kynttilässä on se nenä, mihinkä tullaitetaan. | 9 / 29 |
| Ours | Sitten tehtiin sellaisia pitkiä, niin kuin kynttilä on, tompituisia lankoja, ja sitten pantiin se tikku siihen nenään, niin kuin nyt kynttilässä on se nenä, mihin tuli laitetaan. | 6 / 29 |

**Jaala** · `SKN40a_Jaala_0177`

| | Text | Errors |
|---|---|--:|
| Human | no kuinkas sitten kun lapsi syntyi niin käytiinkös sitä katsomassa saunassa sitten no - käytiinhän sitä katsomassa saunassahan ne syntyi lapset eihän ne kammarissa syntynyt | |
| Whisper | Kun lapsi syntyi, niin käytiinkö sitten katsomassa saunassa? No käytiin ja käytiin ihan sitten katsomassa. Saunassahan ne synty lapset, eihän ne kammariissa syntynyt. | 13 / 24 |
| Ours | No kuinkas sitten, kun lapsi syntyi, niin käytiinkös sitten katsomassa saunassa sitten? No käyttiin, käytiinhän sitten katsomassa, saunassahan ne syntyi lapset, eihän ne kammareissa syntyneet. | 5 / 24 |

**Mikkeli** · `SKN10b_Mikkeli_0024`

| | Text | Errors |
|---|---|--:|
| Human | kuinkas se mämmi oikein tehdään no imelletään ensin jauhot ja keitetään ja sitten paistettiin | |
| Whisper | Kuinka se mämmi oikein tehdään? No imelotetaan ensin ja keitetään ja sit paistetaan. | 5 / 14 |
| Ours | Kuinkas se mämmi oikein tehdään? No ime laitetaan ensin jaot ja keitetään ja sitten paistetaan. | 4 / 14 |

**Mikkeli** · `SKN10b_Mikkeli_0078`

| | Text | Errors |
|---|---|--:|
| Human | ennen kylläkin minäkin olin ahkera marjastamaan ja niitä noukki ja myi ja itselle ja ei siitä nyt tullut mitään ne oli jätettävä kertakaikkisesti kaikki ne työhommat niin no onhan se aikakin | |
| Whisper | Ennen kylläkin minäkin olin ahkera marjastamaan ja niitä noukki ja möi ja iselle ja ei siinä nyt. No oli jätettävä kertaa kaikki ne työhommat. No onhan se aikakin. | 8 / 31 |
| Ours | Ennen kylläkin minäkin oli ahkera marjastamaan ja niitä noutkia möiä isälle, ja ei siinä nyt ollut. Ne oli jätettävä kerta kaikki, kaikki ne työhommat, niin. No onhan se aikakin. | 11 / 31 |

**Rautu** · `SKN15a_Rautu_0104`

| | Text | Errors |
|---|---|--:|
| Human | ja isot paikat niin ei niissä mitenkään se saanut niitä syksyllä puitua mutta meillä kerran viljat alkoivat joutua niin puitiin aina kaksi ahdosta kuitenkin viikossa | |
| Whisper | ja isot paikat, niin ei niissä mitenkään saa niitä syksyllä puitua. Mutta meillä kerran viljat alkoi joutua, niin puitiin aina kaksi ahoista kuitenkin viikossa. | 4 / 25 |
| Ours | Ja isot paikat, niin ei niissä mitenkään se saanut niitä syksyllä puitua, mutta meillä kerran viljat alkoivat joutua, niin puitiin aina kaksi ahoista kuitenkin viikossa. | 1 / 25 |

**Rautu** · `SKN15b_Rautu_0016`

| | Text | Errors |
|---|---|--:|
| Human | me oltiin me tehtiin kaksi löysiä joka vuorokausi kaksi löysiä tehtiin joka vuorokausi kuusi ruplaa joka vuorokausi otettiin rahaa siellä neljätoista viisitoista kopeekkaa vuorokausi tuli maksamaan ruoka ja sajut ja ja mitä oli siellä no äläs sinä huoli | |
| Whisper | Me oltiin, me tehtiin kaksi leusiä joka vuorokausi. Kaksi leusiä tehtiin joka vuorokausi. Kuusi sotlaa joka vuorokausi otettiin rahaa. Siellä 14-15 kopakkaa vuorokausi tuli maksamaan ruoka ja sajuja. Ja oli siellä. No, älä se huoli. | 11 / 38 |
| Ours | Me oltiin, me tehtiin kaksi löysiä joka vuorokauden, kaksi löysiä tehtiin joka vuorokauden, kuusi vuotta joka vuorokauden otettiin rahaa. Siellä nelitoista viisitoista kopokkaa vuorokaudesta tuli maksamaan ruoka ja sadot ja ja oli siellä. No älä sinä huoli. | 10 / 38 |

**Vihti** · `SKN05a_Vihti_0164`

| | Text | Errors |
|---|---|--:|
| Human | mitään juurikasveja ei ollut eikä niitä - - kylvetty elukoille jos oli lanttuakin niin ne kylvettiin johonkin pieneen kun poltettiin joku plätti ja sitten ne istutettiin pieneen plättiin sitten vain taimina | |
| Whisper | Mitään juurikasveja ei ollut eikä niitä ole kylvetty elukoilla. Jos oli lantuuki, niin ne kylvettiin johonkin pieneen, kun poltettiin joku plätti ja sit ne istutettiin pieneen plättiin sit vaan taimina. | 6 / 29 |
| Ours | Mitä juurikasveja ei ollut eikä niitä kylvetty elukoilla, jos oli lanttuukin, niin ne kylvettiin johonkin pieneen, kun poltettiin joku plätti, ja sitten ne istutettiin pieneen plättiin sitten vain taimina. | 3 / 29 |

**Vihti** · `SKN05b_Vihti_0065`

| | Text | Errors |
|---|---|--:|
| Human | jos ne on sitten niin korkealla ollut että ei haulit ole ottaneet tai muuta mutta ei kertaakaan että olisin saanut ja minä olen monta kutia ampunut niitä | |
| Whisper | Jos ne on sitten niin korkealla ollut, ettei haulit ole ottaneet aika muuta, mut ei kertaakaan et olisin saanut ja mä olen monta kutia ampunut niitä. | 6 / 27 |
| Ours | Jos ne on sitten niin korkealla ollut, että ei haulit ole ottaneet tai muuta, mutta ei kertaakaan ne olisi saanut, ja minä olen monta kutia ampunut niitä. | 2 / 27 |

Source: Suomen kielen näytteitä, Kotimaisten kielten keskus (Kotus), via Kielipankki – The Language
Bank of Finland, CC BY 4.0.
