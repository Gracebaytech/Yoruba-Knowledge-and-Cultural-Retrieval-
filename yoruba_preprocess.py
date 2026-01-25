import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from pathlib import Path
from typing import List
import string

# Ensure NLTK stopwords are available
try:
    _ = stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Yoruba stopwords (starter list)
yoruba_stopwords = {
    "mo", "emi", "a", "wa", "iwo", "o", "wọ́n", "òun", "ẹ", "yin", "ara", "ara wọn",
    "eyi", "awon", "àwọn", "yen", "nkan", "ohun", "bee", "bẹ́ẹ̀", "nibi", "ibi", "ibi ti",
    "kan", "kankan", "gbogbo", "kò sí", "diẹ", "eyiti", "gbogbo wọn",
    "ati", "àti", "tàbí", "tabi", "tí", "nígbà", "ṣùgbọ́n", "nítorí", "ṣugbọn", "bi", "bí",
    "gẹ́gẹ́", "sugbon", "síbẹ̀","ní", "ninu", "nínú", "lati", "sí", "si", "pelu", "pẹ̀lú", "sori", "sọ́rí", "lẹ́yìn",
    "lẹ́hin", "nítorí", "kíá", "kí", "ki","jẹ", "máa", "n", "ti", "ń", "ni", "ńṣe", "se", "ṣe", "má", "má ṣe", "le", "ò le",
    "ó ti", "o ti", "kò", "ko", "kò ní", "ó", "ó ń", "yoo", "yóò", "yoò", "yó", "maa",
    "máa ṣe", "nigbati", "nígbà", "nísinsin", "bayi", "báyìí", "tele", "tele naa", "lẹ́yìnna",
    "loni", "lónìí", "ọla", "ní ọjọ́ iwájú", "ṣeese", "ṣéèsẹ́", "lẹ́ẹ̀kan", "nigbakan",
    "nigbakugba",  "kò", "ko", "kii", "kì í", "kìí", "jẹ́jẹ́", "kò sí", "kò lè",
    "se", "ṣe", "njẹ", "njẹ́", "abi", "àbí", "ti", "eyi ti", "eyiti", "àti pé", "pé", "pe", "nipe",
    "o", "e", "ẹ", "eni", "eni tí", "gbogbo", "lọ", "wá", "tun", "tún", "sugbọn",
    "sugbon", "jọ", "jẹ́", "bẹ́ẹ̀ni", "bẹ́ẹ̀kọ́", "tọ́", "nko", "ǹkan", "ṣé", "jẹ́jẹ́", "jẹ́lọ́",
    "gbogbo nkan", "ṣé o", "ṣé ẹ", "jẹ́wọ́", "ọpọlọpọ", "kan naa", "kòkan",
    "gbogbo wọn", "ohun gbogbo", "naa", "ti", "ni", "le", "pe", "o", "ko", "si",
    "wa", "mo", "je", "ki", "yo", "yoo", "nija", "naija", "nini", "nina", "ni",
    "nke", "nkan", "nitori", "lasisu", "nlo", "lolo", "tun", "gbe", "ori", "ile",
    "fun", "wọn", "awọn", "mọ", "ẹni", "ẹyin", "ẹrẹ", "ọdún", "ṣe", "jẹ", "kọ",
    "nọ", "nínú", "náà", "láti", "bí", "pẹ̀lẹ̀", "ọ̀nà", "ọ̀rọ̀", "ọ̀rùn","fi","so","mo","mi","emi","o","e","iwo","a","awa","wa","won","wọn","yin","tiwa",
"tire","temi","tẹni","tawa","tirẹ","tiwọn","ara","ni","ní","je","jẹ","ti",
"tun","maa","ma","ń","n","yoo","yo","yóó","lọ","wa","di","si","lè","le","ati",
"àti","tabi","tàbí","sugbon","ṣùgbọ́n","ṣugbọn","nitori","torí","toripe","bí",
"bi","tí","pé","ati pe","gan","gangan","patapata","rara","rárá","bayi","báyìí",
"bẹẹ","nibẹ","nísinsin","lẹ́sẹ̀kẹsẹ̀","lojoojúmọ́","nigbagbogbo","ohun","nkan",
"ohunkohun","gbogbo","ẹnikan","ẹnikẹ́ni","eyini","eyi","enikeni","iru","abi",
"ábi","o","ó","ò","sí","ṣe","na","nà","nàá","nàà","naa","nàà","nibikibi",
"nibẹ̀","ibìkan","oni","òní","lana","lanaa","lola","òla","lẹ́yìn","lati","láti",
"fún","pẹ̀lú","pelu","níta","nínu","inú","sọdọ","lọ́dọ̀","wá","jẹ́","jẹ̀","jẹ́ẹ̀",
"bákan náà","nítorí náà","a", "aa", "aago", "aare", "aabo", "aago", "aarin", "aati", "aawọ", "aba",
    "abe", "abẹ", "abi", "abièbì", "abiye", "abọ", "abo", "abẹrẹ", "abẹ́",
    "abẹrẹ̀", "abọ̀", "àbọ", "àbí", "àbíyẹ", "abuda", "àbùdá", "abule",
    "àbùlé", "aburu", "àbùrú", "abẹrẹ́", "abíyé", "àdí", "àdì", "ài",
    "àibanuje", "àì", "àìbọ̀", "àìfẹ", "àìlera", "àìlera", "aimọ", "aimokan",
    "àìmò", "àìmọ̀", "aimoye", "àimoyè", "aina", "airi", "àìrí", "airotẹlẹ",
    "àìrọ̀tẹ̀lẹ̀", "aisun", "àìsùn", "aitọ", "àìtó", "aiyede", "aiye", "àyè",
    "àyé", "aìyè", "aiyé", "ajọ", "àjọ", "àjọṣe", "àjọyọ", "àkà", "akọ",
    "àkọ́kọ́", "àkọ́lé", "àkọ́kọ́", "akoko", "àkoko", "àkóko", "àkọ́kọ̀",
    "akokọ", "akori", "àkọrì", "àkórì", "àkọ", "ala", "àla", "àlàáfíà",
    "alala", "àlalà", "alajọṣepọ", "alaini", "alaiṣẹ", "alàìnì", "aláìṣe",
    "aláìsẹ", "alakoso", "alákọso", "alákosọ", "alabapadẹ", "aladugbo",
    "aládùgbò", "alagbara", "alágbàra", "alakoso", "alákòsọ", "alaanu",
    "aláànù", "alakosọ", "alakosọ", "alaye", "àláyè", "ale", "àlé", "àlẹ́",
    "aligbawi", "àlígbàwì", "alọ", "àlọ", "alade", "àládé", "alafọ", "alafia",
    "alaafia", "àлаáfíà", "alaigbọran", "àláìgbọran", "alaigbọ́", "alaiṣesi",
    "alaiyẹ", "alaiye", "alaiyẹ", "aláìyẹ", "àlẹ́", "àlẹjo", "àlẹjọ̀",
    "aletan", "àlétan", "alẹ́mọ", "alemọ", "alẹ́", "alẹ", "alẹni", "aleni",
    "alénì", "àlẹnì", "alẹ́gbẹ", "alẹ́gbẹ̀", "alẹ́gbẹ́", "alẹ́jọ", "alẹ́jẹ",
    "alọwọ", "alọwọ̀", "alowọ", "aláìdá", "aládùúgbò", "aládùrú", "aladodo",
    "aládódo", "alatilé", "alátìlẹ́yìn", "alatilẹyìn", "alatilẹ", "alate",
    "alatẹ", "àlà", "àlàọdé", "àlàáfíà", "àláfíà", "àlùfáà", "àlùfáà",
    "ama", "àmá", "àmà", "àmáà", "amaa", "a maa", "àmàà", "amáà", "amani",
    "amánì", "amọ", "àmọ̀", "àmọ́", "amọ", "àmọ̀", "amala", "àmálà", "amala",
    "amunisin", "àmúnìsìn", "àǹfàánì", "anfaani", "anfáánì", "ani", "àni",
    "àní", "ànì", "aní", "anì", "anikan", "àníkan", "aniye", "aǹiye", "anu",
    "ànu", "ànú", "ànù", "ànúre", "ànùré", "anure", "a nure", "ara", "àra",
    "àrà", "àrá", "araa", "arakunrin", "arakunrìn", "arabinrin", "arábìnrin",
    "arin", "àrín", "àrínkànkàn", "àrí", "àrì", "arí", "arì", "aro", "àro",
    "àrò", "àró", "arosọ", "àròsọ̀", "arọsọ", "arọ", "àrọ́", "aru", "àru",
    "àrù", "àrú", "aarin", "ààrin", "ààrin", "àárin", "ati", "àtí", "àtì",
    "atì", "atìgbè", "atígbà", "atìgbà", "a ti", "ato", "àto", "àtó", "àtò",
    "a tò", "àtọ", "àtọ́", "àkọ́tọ̀", "atọka", "àtọ́kà", "àtọ̀kà", "atiẹ",
    "àtiẹ", "atìsì", "àtìsì", "atiwọn", "àtiwọn", "àtìwọn", "atùpale",
    "àtùpalè", "àtùnṣe", "atunṣe", "àtúnṣẹ", "atupa", "àtúpà", "àtúpa",
    "a tun", "àtun", "awa", "àwá", "àwà", "àwàrà", "àwán", "awán", "awawi",
    "àwàwì", "awon", "àwọn", "àwọ̀n", "àwọ́n", "àwọ̀nnà", "àwọ́nnà", "àwọnnà",
    "ayafi", "àyáfì", "àyàfì", "ayafì", "aya", "àya", "àyà", "àyá", "aye",
    "àye", "àyè", "àyẹ", "àyé", "àyẹ́", "ayo̩", "ayọ", "àyọ̀", "ayo", "àyo",
    "àyó", "àyò", "àyóò", "àyòò", "ayẹ́", "ayé", "ayi", "àyí", "àyì", "àyi",
    "àyìn", "àyú", "àyù", "àyún", "ayaba", "àyàbà", "aaya", "ààyà", "ààya",
    "àáya", "ayika", "ayikà", "ayiká", "ayékun", "ayékùn", "ayetoro",
    "àyétòrò", "ayẹtọ", "ayika", "ayidayida", "àyìdàyidà", "ayipada",
    "ayipadà", "àyípadà", "ayipadá", "ayikà", "ayiká", "b", "ba", "bà", "bá",
    "bàa", "báà", "bàà", "bàa", "ba bọ", "báwọ̀", "bawọ", "bawo", "báwo",
    "báwọ", "be", "bẹ", "bẹ̀", "bẹ", "bẹ́ẹ̀", "bẹẹ", "bẹ̀ẹ̀", "bẹẹ́", "bẹ́ẹ́",
    "bẹẹ", "bi", "bí", "bì", "bì", "bí", "bíi", "bii", "bíi ṣe", "biṣe",
    "bẹẹni", "bẹ́ẹ̀nì", "bẹ́ẹni", "bẹẹni", "bẹẹkọ", "bọ", "bọ", "bọ̀", "bọ́",
    "bọ̀", "bori", "bọri", "bọ́rẹ", "bọsẹ", "bọsẹ̀", "bọ̀sẹ̀", "bọ sẹ", "bu",
    "bù", "bú", "bọla", "bàbá", "baba", "bàbà", "bàbàlórìṣà", "bẹni",
    "bẹẹni", "bẹkọ", "bẹjọ", "bẹẹkọ", "bẹẹkọ", "bẹẹkọọ", "bẹẹkọọ", "bẹ̀ẹrẹ̀",
    "bàjẹ́", "bájẹ", "bajẹ", "baje", "d", "da", "dà", "dá", "dàà", "dáà",
    "dáa", "dàa", "dara", "dára", "dàra", "daraa", "dẹ", "dẹ", "dẹ̀", "dẹ̀",
    "dẹ́", "dẹ́", "de", "dè", "dé", "déédé", "di", "dẹ́kun", "dẹkun", "dinku",
    "dínkù", "dín", "dínkù", "dípò", "dìpò", "dipò", "dípò", "dì", "do",
    "dò", "dó", "dòò", "dodo", "dọ", "dọ̀", "dọ́", "e", "ẹ", "è", "è", "é",
    "é", "ẹni", "ẹ̀ni", "eni", "ènì", "enìkan", "ẹnikan", "ẹnikọọkan",
    "ẹnikọọ̀kan", "eni ti", "ẹni tí", "ẹniti", "ẹni tí", "ẹyin", "èyìn",
    "eyi", "eyì", "eyí", "èyí", "eyi tí", "eyi to", "eya", "ẹya", "ẹmu",
    "ẹ̀mu", "ẹmi", "emi", "èmì", "ẹ̀mì", "ẹniyan", "eeyan", "eeyan", "eeyan",
    "ẹrù", "ẹ̀rù", "ẹsẹ", "ẹsẹ̀", "ẹsẹ́", "èsẹ̀", "ese", "esin", "ẹsin",
    "ẹ̀sìn", "esin", "ẹ̀sì", "esì", "esin", "ẹlòmíràn", "elomiran", "elomi",
    "ẹlomi", "ẹlòmíì", "g", "ga", "gà", "gá", "gàà", "gáà", "gba", "gbá",
    "gbà", "gbaa", "gbàà", "gbà", "gbá", "gbọ", "gbọ́", "gbọ̀", "gbọ", "gbọ̀n",
    "gbọn", "gbàjẹ́", "gbele", "gbẹ̀lẹ̀", "gbẹ̀yin", "gbeyin", "gbiyanju",
    "gbìmọ̀", "gbímọ̀", "gbàgbọ́", "gbagbọ", "gbagbe", "gbagbé", "gbàgbé",
    "gbe", "gbé", "gbè", "gbé", "gun", "gún", "gùn", "gúnre", "gúnrè", "gúnré",
    "h", "ha", "hà", "há", "han", "hàn", "hán", "han-an", "ho", "hò", "hó",
    "hu", "hú", "hù", "hurú", "hùrú", "i", "ì", "í", "ì", "í", "iba", "ibà",
    "ibá", "ibàyé", "ibe", "ibi", "ibì", "ibí", "ìbí", "ibìkan", "ibi kan",
    "ibikan", "ibo", "ibò", "ibó", "ibi ti", "ibi tí", "idi", "ìdì", "ìdí",
    "ìdí", "idojukọ", "idojú", "idi ti", "idibajẹ", "ìdí le", "ìdílé",
    "ìdílé rẹ", "ifẹ", "ìfẹ́", "ifẹ̀", "ìfẹ̀", "ifẹni", "ìfẹ́ni", "igba",
    "igbà", "igbá", "igbe", "ìgbé", "ìgbà", "igbani", "ìgbàní", "ìgbàgbọ́",
    "igbagbọ", "igboro", "ìgboro", "igbogbo", "ìgbogbo", "igbagbogbo",
    "igbagbogbò", "ìgbàpọ̀", "ìgbẹ́run", "igbagbogbo", "ìgbẹ̀yà", "ìgbọ̀nsẹ̀",
    "ìgbọ́n", "ijo", "ìjọ", "ìjọba", "ijọba", "ìjọgbọ́n", "ìjọsí", "ikasI",
    "ìká", "ika", "ike", "ìkẹ̀", "ìké", "ìkẹ́ta", "ikẹhin", "ikẹhin", "ìkẹ́hìn",
    "ikanju", "ikú", "iku", "ìkú", "ikọ̀", "ìkọ̀", "ikọ", "ikọsẹ", "ilẹ",
    "ilé", "ile", "ilé-ẹkọ", "ìlú", "ilu", "ìlú̀", "ilana", "ìlànà", "ìlànàa",
    "ilepa", "ìlépa", "ilẹ̀kùn", "ilẹkun", "ìlẹ̀kejì", "ìlẹ́kẹ̀jì", "imọ́",
    "ìmọ̀", "ìmọ̀lára", "ìmọ̀tótó", "inú", "inu", "inú", "ìnira", "inira",
    "ìnìkan", "inakan", "inú rẹ", "inu rẹ̀", "inú ti", "inú mi", "iri", "ìrì",
    "ìrẹ̀kọjá", "ìrẹ́kọjá", "irọrun", "iro", "iró", "ìrò", "ìrọ̀", "ìrọ̀lẹ́",
    "ìrìn", "iriri", "ìrírí", "ìrìnàjò", "ìrìnàjẹ̀", "itan", "ìtàn", "ìtànná",
    "ìtànṣe", "ìtọ́sọ́nà", "ìtọ́sọnà", "ìtọ́kà", "ìtọ́ka", "ìtòsí", "itan",
    "ìtàn", "ìtànná", "ìtàn", "ìtọ́sọ", "ìtọ́sọ́nà", "ìtànkosẹ̀",
    "ìtànkànkàn", "jẹ", "je", "jẹ̀", "jẹ́", "jẹjẹ", "jẹ̀jẹ̀", "jẹ́jẹ́",
    "jẹkẹta", "jẹkẹ̀tà", "jẹ́ ki", "jẹ kí", "jẹgbẹ", "jẹgbẹ̀", "jẹ́sọ", "jọ",
    "jọ̀", "jọ́", "jọpọ", "pọ̀", "pọ", "pọ́", "pọ̀ọ́", "pọ̀ọ̀", "jẹwọn",
    "jẹ́wọ̀n", "jẹ", "ji", "jí", "jì", "jì", "jẹmọ", "jẹmọ̀", "jọwọ", "jọ̀wọ́",
    "jọwọ́", "k", "ka", "ká", "kà", "kà", "káàkiri", "kaakiri", "kan", "kan",
    "kàn", "kán", "kọ", "kò", "kọ́", "kọ̀", "kọjá", "kọjá", "kọjà", "kọ́jà",
    "kọ́", "kọ̀ọ́kan", "ko", "kò", "kó", "kò sí", "ko si", "kò sí", "kòtò",
    "kòtọ̀", "kọ̀tọ̀", "kọtọ", "kọ́rọ̀", "kọrọ̀", "kọsẹ", "kọsẹ̀", "kọ́sẹ̀",
    "kọlọ", "kàn", "kànkàn", "ki", "kí", "kì", "kí", "kí ni", "kilode",
    "kí ni", "kí nìdí", "kí ni idi", "kini", "kìnìun", "kìnìún", "kíyèsi",
    "kiyesi", "kíyè̀sí", "l", "la", "là", "lá", "làà", "láà", "láa", "lẹ",
    "le", "lè", "lẹ", "lẹ̀", "lèé", "lẹ́ẹ̀kan", "lẹ́kan", "lẹ́ẹ̀kọ̀ọ̀kan",
    "lẹẹkan", "lẹ́sẹ̀kẹsẹ̀", "làyìn", "láyìn", "lára", "lara", "làrà", "lẹ́nu",
    "lenu", "lẹnu", "lẹ́nù", "lẹ́yìn", "lẹ́yin", "lẹ́hìn", "lẹ́gbẹ̀ẹ́",
    "lẹgbẹẹ", "lẹgbẹ", "lẹgbẹ̀", "lo", "lọ", "lọ́", "lọ̀", "lọpọlọpọ", "lọna",
    "lọ́nà", "lọ́nàa", "lọ́nà náà", "lọ́nà yẹn", "lọlẹ", "lọ́ọ̀rọ̀", "lọ́ọ̀",
    "lọ si", "lọ sí", "lọ ní", "lori", "lórí", "lóríi", "láti", "lati", "làtì",
    "latì", "làti", "latí", "látì", "lọ́pọ̀", "lọpọ", "lọ́pọ̀lọpọ̀", "m", "ma",
    "mà", "má", "màà", "máa", "màa", "maa", "màà", "máà", "máa", "maṣe",
    "maṣè", "máṣe", "màṣe", "màjẹ̀mú", "maa ṣe", "mi", "mì", "mí", "míì",
    "mìíràn", "miiran", "mọ", "mọ̀", "mọ́", "mo", "mò", "mó", "mọ̀nà", "mọna",
    "mọ̀kan", "mọkan", "mọ̀kanlá", "mọ́ni", "mu", "mú", "mù", "mú", "mùú",
    "múlẹ̀", "munadoko", "mùnadokò", "n", "na", "nà", "ná", "nàà", "náà",
    "naa", "nitori", "nitorí", "nitorì", "nitori pe", "nítorí", "nítorí pé",
    "nítorí náà", "ni", "ní", "nì", "níí", "nìkan", "nikan", "nikàn",
    "nikan náà", "nile", "nílé", "nilee", "nílẹ̀", "nilè", "níbẹ̀", "nibẹ",
    "nibẹ̀", "nibi", "níbí", "nìbí", "nibikibi", "nibikan", "nígbà", "nigbà",
    "nígbà tí", "nígbà náà", "nígbà gbogbo", "nígba naà", "nííṣe", "niṣe",
    "nìṣè", "níṣè", "níṣé", "níṣe", "níba", "níba", "níbo", "nibò", "nibó",
    "níbo", "nínú", "ninu", "ninù", "ninú", "nira", "nìrà", "nirà", "o", "ọ",
    "ò", "ó", "òò", "óò", "óó", "oorun", "ọjọ", "ọjọ̀", "ọjọ́", "ọ̀sẹ̀", "ọsẹ",
    "ọdun", "ọdun", "ọdún", "o dun", "òun", "oun", "ọ̀nà", "ona", "ọna", "odo",
    "odò", "odó", "odindi", "ọdindi", "òdìí", "odi", "odì", "òdì", "òdìkejì",
    "oke", "okè", "oké", "òkè", "okè", "oko", "ọkọ", "ọkọ̀", "ọkọ́", "okọ",
    "ọkọọkan", "ọkọọkan", "ọkọ̀ọ̀kan", "ọkọ̀ó", "ọ̀rọ̀", "ọrọ", "ọ̀rọ̀̀",
    "ọ̀rọ̀ náà", "ọ̀rọ̀ tí", "o si", "o sì", "o sì ni", "ó", "ó sì", "ó sì ní",
    "ó ti", "oti", "òtítọ́", "otito", "òtítọ̀ ni", "òtítọ̀́ ni", "òótọ́",
    "òótọ̀", "otitọ", "òun", "oun", "ọhun", "ohun", "ọ̀hún", "ọ̀kan", "ọkan",
    "ọkan si", "ọkaǹ", "ọwọ", "ọwọ́", "owo", "ọ̀pọ̀", "pọ̀", "pọ̀lọ́pọ̀",
    "ọ̀pọ̀lọpọ̀", "pọ̀ọ̀", "pọ̀ọ́", "pọ̀pọlọpọ", "ọ̀pọ̀lọ̀pọ̀", "ojo", "ojò",
    "ojó", "òde", "òdè", "òdè yìì", "òdè yii", "òdè lọ", "òdè òní", "p", "pa",
    "pà", "pá", "pàà", "páà", "pe", "pé", "pè", "pẹ́", "pelu", "pẹ̀lú", "pọ",
    "pọ̀", "pọ́", "pọ̀ọ́", "pọ̀lọ́pọ̀", "pọ̀lọ̀pọ̀", "pọ̀pọ̀", "r", "ra", "rà",
    "rá", "ràa", "ráa", "rárá", "rara", "rọ", "rọ", "rò", "ró", "ròpò",
    "ropò", "rọ̀run", "rorun", "ròrùn", "rìn", "rìn", "rin", "rí", "rì", "rí",
    "ríì", "rìì", "ríí", "rẹ", "re", "rẹ̀", "rẹ̀", "rẹ̀ ní", "rẹ̀ ni", "ri mi",
    "rọ̀", "rò mọ́", "ròmò", "s", "sa", "sà", "sá", "sàaà", "saa", "sáa",
    "ṣe", "ṣè", "ṣé", "ṣepe", "ṣépe", "ṣe ni", "ṣè nì", "ṣèṣe", "se", "sè",
    "sé", "si", "sí", "sì", "síi", "síbẹ̀", "sibẹ", "síbẹ̀", "sibẹ̀", "sibẹ",
    "sibi", "síbí", "síbí kan", "sibikan", "síbìkan", "si i", "sí i", "sibe",
    "sí bè", "sugbon", "ṣùgbọ́n", "ṣùgbọ́n", "ṣugbọn", "sibẹ", "sibẹ̀sibẹ̀",
    "ṣugbọn", "ṣùgbọ́n", "t", "ta", "tà", "tá", "tàà", "taa", "ti", "tí",
    "tì", "tíí", "tii", "tíì", "tíì", "tìkẹ̀sẹ̀", "tẹẹsẹ̀", "tùn", "tun",
    "tún", "tún", "tàn", "tan", "tani", "tanì", "taní", "ta ló", "ta ni",
    "tani lo", "to", "tó", "tò", "tó", "tori", "torí", "torí pé", "toripe",
    "títí", "titi", "titi di", "títí di", "títí tí", "tàpàpà", "tán-an",
    "tó sì", "tò sì", "tò lọ", "tò lọ", "u", "wọ", "wọ̀", "wọ́", "wọ́n", "wón",
    "wọn", "wá", "wa", "wà", "wà", "were", "wèrè", "wèrè", "wáà", "wáa",
    "wàà", "wá sí", "wà ní", "wà ní", "we", "wẹ", "wẹ̀", "wẹ́", "wɔ", "wọ̀",
    "wọ́", "wọ́n", "wọ́n ni", "wọ́n sì", "wọ́n wí", "wọ́n sọ", "wọ́n gbà",
    "wọ́n jẹ", "y", "ya", "yà", "yá", "yàà", "yaa", "yán", "yàn", "yán-an",
    "yan-an", "yi", "yìí", "yìí ni", "yí", "yíí", "yíí", "yìì", "yii", "yì",
    "yìn", "yìn-in", "yẹn", "yẹn ni", "yẹn tí", "yá", "yá", "yin", "yìn",
    "yà", "yà", "yoo", "yòò", "yóò", "yóó", "yọ", "yọ̀", "yọ́", "ṣ", "ṣa",
    "ṣá", "ṣà", "ṣàá", "ṣaa", "ṣaaju", "ṣaajù", "ṣaajú", "ṣájù", "ṣaduro",
    "ṣadùró", "ṣagbè", "ṣagbé", "ṣàbì", "ṣe", "ṣè", "ṣé", "ṣepọ", "ṣepọ̀",
    "ṣeéṣe", "ṣe e", "ṣeni", "ṣeni tí", "ṣeni to", "ṣi", "ṣì", "ṣí", "ṣii",
    "ṣí", "ṣígọ", "ṣigọ", "ṣòro", "ṣoro", "ṣòrọ̀", "ṣọ", "ṣọ̀", "ṣọ́", "ṣọ̀kan",
    "ṣọkan", "ṣọ̀kan si", "ṣọ̀kan",'sọ','wo', 'á','bo','wi','w','v','fe', 'ri','pẹlu',
    'lọwọ','jowo','jẹ́','jẹ','ń','ńṣe','ńṣe','ńkọ́','ńkọ́','ńjẹ́','ńjẹ́','ló','nípa','yín','nnkan','lowo','fẹ',
    'yẹ','un','ge','nla','ye','já','fa','nii','daadaa','ree','ke','ṣẹ','ku','nigba','fẹẹ','nipa','lawọn','gẹgẹ','sportsinyorubagmailcom',
    'httpwwwtwittercomsportsinyoruba','ẹọ','pẹlú', 'wipe','ṣl','nipa'

}

eng_stopwords = set(stopwords.words('english'))
custom_stopwords = eng_stopwords.union(yoruba_stopwords)


def decontractions(phrase: str) -> str:
    """Expand common English contractions found in mixed text.

    This is a simple rule-based normalization useful when English contractions
    appear in code-switched data.
    """
    if not isinstance(phrase, str):
        return phrase
    phrase = re.sub(r"won['’]t", "will not", phrase)
    phrase = re.sub(r"can['’]t", "can not", phrase)
    phrase = re.sub(r"n['’]t", " not", phrase)
    phrase = re.sub(r"['’]re", " are", phrase)
    phrase = re.sub(r"['’]s", " is", phrase)
    phrase = re.sub(r"['’]d", " would", phrase)
    phrase = re.sub(r"['’]ll", " will", phrase)
    phrase = re.sub(r"['’]ve", " have", phrase)
    phrase = re.sub(r"['’]m", " am", phrase)
    return phrase


def clean_ocr_errors(text: str) -> str:
    """Apply common OCR correction heuristics for Yoruba/code-switched texts.

    This function performs lightweight, reversible fixes such as:
    - common digit/character confusions (0->ọ, 3->ẹ, 2->ṣ, etc.)
    - remove hyphenation at line endings (word-\nnext -> wordnext)
    - replace ligatures and unusual unicode artifacts
    - strip control characters and repeated punctuation
    """
    if not isinstance(text, str):
        return text

    # Normalize some unicode ligatures and weird characters
    ligature_map = {
        '\uFB01': 'fi',  # ﬁ
        '\uFB02': 'fl',  # ﬂ
        '\u2019': "'",
        '\u2018': "'",
        '\u2013': '-',
        '\u2014': '-',
        '\u00A0': ' ',
    }
    for k, v in ligature_map.items():
        text = text.replace(k, v)

    # Common OCR character confusions observed in Yoruba datasets
    fixes = {
        '0': 'ọ',  # zero mistaken for open-o
        '1': 'l',  # one mistaken for lowercase L or I
        '3': 'ẹ',
        '2': 'ṣ',
        '\\|': 'l',  # vertical bar mistaken for l
        '\\u201C': '"',
        '\\u201D': '"',
    }
    for wrong, right in fixes.items():
        try:
            text = re.sub(wrong, right, text)
        except re.error:
            # if the pattern isn't a regex, do a simple replace
            text = text.replace(wrong, right)

    # Remove hyphenation at line breaks: word-\nnext -> wordnext
    text = re.sub(r"-\s*\n\s*", '', text)
    text = re.sub(r"[^\w\s]", "", text) # Remove non-word characters


    # Remove stray control characters
    text = ''.join(ch for ch in text if ch.isprintable())

    # Collapse multiple punctuation or whitespace
    text = re.sub(r"[ ]{2,}", ' ', text)
    text = re.sub(r"[\.]{3,}", '...', text)

    return text


def preprocess_yoruba(text: str,
                     lowercase: bool = True,
                     remove_extra_spaces: bool = True,
                     keep_yoruba_only: bool = True) -> str:
    """Preprocess a single Yoruba text string.

    Steps:
    - Unicode normalize (NFC)
    - Remove non-printable characters
    - Optionally lowercase
    - Expand some contractions
    - Remove extra whitespace
    - Optionally keep only Yoruba characters and a small set of punctuation
    - Remove stopwords (English + Yoruba list)

    Returns the cleaned text (string). If input is not a string, returns empty string.
    """
    if not isinstance(text, str):
        return ""

    # Unicode normalize then apply OCR cleaning heuristics early
    text = unicodedata.normalize('NFC', text)
    text = clean_ocr_errors(text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation + "”“\"''‘’–—…"))

    if lowercase:
        text = text.lower()

    text = decontractions(text)

    if remove_extra_spaces:
        text = re.sub(r'\s+', ' ', text).strip()

    if keep_yoruba_only:
        yoruba_chars = 'abcdefghijklmnopqrstuvwxyzẹọṣáéíóúàèìòùABCDEFGHIJKLMNOPQRSTUVWXYZẸỌṢÁÉÍÓÚÀÈÌÒÙ'
        allowed_punct = ' .,!?:;\'"--()[]{}…‘’“”–—'
        text = ''.join(c for c in text if c in yoruba_chars or c in allowed_punct)

    tokens = text.split()
    tokens = [t for t in tokens if t not in custom_stopwords]
    return " ".join(tokens)


def preprocess_list(text_list: List[str]) -> List[str]:
    """Apply preprocess_yoruba to a list of strings."""
    return [preprocess_yoruba(t) for t in text_list]


__all__ = [
    'preprocess_yoruba',
    'preprocess_list',
    'decontractions',
    'custom_stopwords',
    'yoruba_stopwords',
    'clean_ocr_errors'
]


# Module intended for import in model-building pipelines; no top-level execution code.
