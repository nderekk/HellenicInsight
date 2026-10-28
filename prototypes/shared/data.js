/* Shared synthetic sample data for the front-page prototypes.
   Illustrative events and headlines, not real coverage. Each article: [outlet, score 0–1, uncertainty, headline]. */
window.PB = (() => {
  const OUTLETS = ["efsyn.gr","thepressproject.gr","in.gr","tanea.gr","ertnews.gr","naftemporiki.gr","iefimerida.gr","protothema.gr"];
  const EVENTS = [
    {id:"opekepe",time:"08:12",topic:"pol",el:"Βουλή: ψηφίστηκε η εξεταστική για τον ΟΠΕΚΕΠΕ με 158 ψήφους",en:"Parliament approves inquiry committee into OPEKEPE subsidies",
     arts:[["efsyn.gr",.18,.06,"«Η εξεταστική ανοίγει τον φάκελο που η κυβέρνηση ήθελε κλειστό»"],["thepressproject.gr",.22,.08,"Ποιοι πήραν τις επιδοτήσεις: τα ονόματα πίσω από τον ΟΠΕΚΕΠΕ"],["efsyn.gr",.29,.09,"Οι αγρότες που περιμένουν ακόμη τις πληρωμές τους"],["in.gr",.36,.10,"Με 158 ψήφους η σύσταση εξεταστικής επιτροπής"],["tanea.gr",.40,.11,"Τι θα εξετάσει η επιτροπή και ποιοι θα καταθέσουν"],["ertnews.gr",.49,.05,"Εγκρίθηκε από την Ολομέλεια η εξεταστική επιτροπή"],["naftemporiki.gr",.55,.07,"Τι σημαίνει η εξεταστική για τις αγροτικές πληρωμές"],["iefimerida.gr",.68,.09,"Κυβέρνηση: «Δεν έχουμε τίποτα να κρύψουμε»"],["protothema.gr",.74,.07,"Η αντιπολίτευση στήνει σκηνικό πόλωσης με την εξεταστική"],["iefimerida.gr",.79,.08,"Μητσοτάκης: διαφάνεια μέχρι τέλους"],["protothema.gr",.86,.10,"Τα λάθη της προηγούμενης διοίκησης του Οργανισμού"]]},
    {id:"gsee",time:"09:40",topic:"soc",el:"Απεργία ΓΣΕΕ: χωρίς ΜΜΜ η Αθήνα την Πέμπτη",en:"GSEE strike: no public transport in Athens on Thursday",
     arts:[["efsyn.gr",.12,.05,"Όλοι στους δρόμους: γιατί απεργούν οι εργαζόμενοι"],["thepressproject.gr",.15,.07,"Η απεργία που η κυβέρνηση δεν θέλει να δει"],["in.gr",.33,.09,"Πώς θα κινηθούν τα ΜΜΜ την Πέμπτη"],["tanea.gr",.38,.10,"Στάσεις εργασίας και συλλαλητήρια σε όλη τη χώρα"],["ertnews.gr",.47,.04,"Χωρίς μετρό, ηλεκτρικό και τραμ την Πέμπτη"],["naftemporiki.gr",.53,.06,"Το κόστος της απεργίας για τις επιχειρήσεις"],["ertnews.gr",.50,.05,"Ποια δρομολόγια λεωφορείων θα εκτελεστούν"]]},
    {id:"asylum",time:"11:05",topic:"pol",el:"Μεταναστευτικό: νέα ρύθμιση για τις αιτήσεις ασύλου στην Κρήτη",en:"Migration: new rules for asylum applications in Crete",
     arts:[["efsyn.gr",.16,.07,"Οργανώσεις: η ρύθμιση παραβιάζει το διεθνές δίκαιο"],["thepressproject.gr",.20,.08,"Στον Νότο της Κρήτης, οι άνθρωποι πίσω από τους αριθμούς"],["in.gr",.35,.10,"Τι αλλάζει στις αιτήσεις ασύλου"],["ertnews.gr",.52,.05,"Η νέα διαδικασία για τις αιτήσεις στην Κρήτη"],["naftemporiki.gr",.58,.06,"Τι προβλέπει το σχέδιο για τα κέντρα υποδοχής"],["iefimerida.gr",.66,.08,"Αυστηρά μέτρα για τις ροές νότια της Κρήτης"],["iefimerida.gr",.71,.08,"Υπουργός: η Ελλάδα φυλάει τα σύνορα της Ευρώπης"],["protothema.gr",.75,.09,"Κρήτη: οι κάτοικοι ζητούν λύση τώρα"],["protothema.gr",.90,.11,"«Πύλη εισόδου» η Κρήτη: ο χάρτης των ροών"]]},
    {id:"oil",time:"12:30",topic:"eco",el:"Ακρίβεια: σε ιστορικό υψηλό η τιμή του ελαιολάδου στα ράφια",en:"Cost of living: olive oil shelf prices at a record high",
     arts:[["ertnews.gr",.48,.05,"Πόσο κοστίζει σήμερα το λίτρο ελαιόλαδο"],["naftemporiki.gr",.56,.06,"Ελαιόλαδο: οι εξαγωγές πιέζουν τις τιμές"],["iefimerida.gr",.67,.09,"Μέτρα στήριξης για τους καταναλωτές εξετάζει η κυβέρνηση"],["protothema.gr",.72,.08,"Τα κυκλώματα που ανεβάζουν τις τιμές"],["iefimerida.gr",.81,.09,"Η αγορά σταθεροποιείται, λέει το υπουργείο"]]},
    {id:"energy",time:"13:55",topic:"pol",el:"Δήλωση πρωθυπουργού για το ενεργειακό νομοσχέδιο",en:"Prime Minister's statement on the energy bill",
     arts:[["efsyn.gr",.14,.06,"Νομοσχέδιο για τους λίγους, λογαριασμός για τους πολλούς"],["thepressproject.gr",.24,.08,"Ποιος κερδίζει από το νέο ενεργειακό πλαίσιο"],["tanea.gr",.41,.10,"Τι αλλάζει στους λογαριασμούς ρεύματος"],["ertnews.gr",.50,.04,"Ο πρωθυπουργός για το ενεργειακό νομοσχέδιο"],["naftemporiki.gr",.60,.07,"Οι αγορές υποδέχονται θετικά το νομοσχέδιο"],["iefimerida.gr",.77,.08,"Μητσοτάκης: φθηνότερο ρεύμα από το 2027"],["protothema.gr",.84,.09,"Η αντιπολίτευση χωρίς πρόταση για την ενέργεια"]]},
    {id:"varnavas",time:"14:12",topic:"soc",el:"Πυρκαγιά στη Βαρνάβα: ξεκινούν οι αποζημιώσεις για τους πληγέντες",en:"Varnavas fire: compensation payments begin",
     arts:[["ertnews.gr",.49,.04,"Ανοίγει η πλατφόρμα για τις αποζημιώσεις"],["naftemporiki.gr",.57,.06,"Το ύψος των ενισχύσεων ανά νοικοκυριό"],["iefimerida.gr",.70,.08,"Ταχύτατη η αποζημίωση των πληγέντων"],["protothema.gr",.78,.09,"Οι πρώτες πληρωμές φτάνουν στη Βαρνάβα"]]},
    {id:"bank",time:"14:18",topic:"eco",el:"Χρηματιστήριο: άνοδος στις τράπεζες μετά τα αποτελέσματα τριμήνου",en:"Athens Stock Exchange: banks rise after quarterly results",
     arts:[["naftemporiki.gr",.56,.05,"Ράλι στις τράπεζες μετά τα αποτελέσματα"],["ertnews.gr",.50,.05,"Κέρδη στο Χρηματιστήριο Αθηνών"]]}
  ];
  const T = {
    el:{date:"Τρίτη 28 Οκτωβρίου",events:"Γεγονότα",blind:"Τυφλά σημεία",method:"Μεθοδολογία",updated:"Ενημέρωση",
        L:"Αριστερά",C:"Κέντρο",R:"Δεξιά",Ls:"Αριστ.",Cs:"Κέντρο",Rs:"Δεξιά",
        articles:n=>n===1?"1 άρθρο":`${n} άρθρα`,outlets:n=>n===1?"1 πηγή":`${n} πηγές`,
        none:s=>`Καμία κάλυψη από ${s==="L"?"τα Αριστερά":s==="R"?"τα Δεξιά":"το Κέντρο"}`,
        basis:"Οι αριθμοί μετρούν άρθρα, όχι ψήφους. Θέση ανά άρθρο, με αβεβαιότητα (±).",
        sample:"Δείγμα δεδομένων: όχι πραγματική κάλυψη",sampleShort:"Δείγμα δεδομένων",counted:"Καταμετρήθηκε 14:20 από 8 πηγές",
        open:"Δείτε το γεγονός",first:"Πρώτη αναφορά",topics:{pol:"Πολιτική",soc:"Κοινωνία",eco:"Οικονομία"},
        dayTitle:"Η μέρα μέχρι τώρα",blindLede:"Γεγονότα που μια πλευρά δεν κάλυψε καθόλου.",all:"Όλα"},
    en:{date:"Tuesday 28 October",events:"Events",blind:"Blindspots",method:"Methodology",updated:"Updated",
        L:"Left",C:"Center",R:"Right",Ls:"Left",Cs:"Center",Rs:"Right",
        articles:n=>n===1?"1 article":`${n} articles`,outlets:n=>n===1?"1 outlet":`${n} outlets`,
        none:s=>`No coverage from the ${s==="L"?"Left":s==="R"?"Right":"Center"}`,
        basis:"Figures count articles, not votes. Each article is placed on its own, with uncertainty (±).",
        sample:"Sample data: not real coverage",sampleShort:"Sample data",counted:"Counted 14:20 from 8 outlets",
        open:"Open the event",first:"First reported",topics:{pol:"Politics",soc:"Society",eco:"Economy"},
        dayTitle:"The day so far",blindLede:"Events one side of the spectrum didn't cover at all.",all:"All"}
  };
  const side = s => s<=.40 ? "L" : s<=.60 ? "C" : "R";
  const count = e => { const c={L:0,C:0,R:0}; e.arts.forEach(a=>c[side(a[1])]++); return c; };
  const outlets = e => new Set(e.arts.map(a=>a[0])).size;
  const blind = e => { const c=count(e); return e.arts.length>=4 ? ["L","R"].filter(s=>c[s]===0) : []; };
  const esc = s => String(s).replace(/[&<>"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[ch]));
  const short = o => o.replace(/\.gr$/,"");
  const head = (e,lang) => lang==="el" ? e.el : e.en;
  const sub = (e,lang) => lang==="el" ? e.en : e.el;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches || new URLSearchParams(location.search).has("still");
  if (still) document.documentElement.classList.add("still");
  const initialLang = new URLSearchParams(location.search).get("lang")==="en" ? "en" : "el";
  return {OUTLETS,EVENTS,T,side,count,outlets,blind,esc,short,head,sub,still,initialLang};
})();
