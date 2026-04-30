import React, { useState, useEffect, createContext, useContext } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ShieldCheck, Activity, Scale, Search, FileText, Orbit, LayoutGrid, Clock, ChevronLeft, ChevronRight, Globe } from "lucide-react"

// ==========================================
// --- 1. LANGUAGE & TRANSLATION SYSTEM ---
// ==========================================
type Language = 'en' | 'el';
const LanguageContext = createContext<{ lang: Language, setLang: (l: Language) => void }>({ lang: 'en', setLang: () => {} });

const t = {
  en: {
    navAnalyzer: "Analyzer",
    navLiveFeed: "Live Database",
    heroSub: "Paste a Greek news article below for instant AI analysis.",
    analyzeBtn: "Analyze",
    recentTitle: "Recent Analyses",
    noRecent: "No recent analyses found. Waiting for database sync...",
    liveFeedTitle: "Live Pipeline Feed",
    waitingData: "Waiting for Spark pipeline to insert documents into MongoDB...",
    prev: "Prev",
    next: "Next",
    page: "Page",
    of: "of",
    analysisVerified: "Analysis Verified",
    source: "Source:",
    noTags: "No tags available",
    polLean: "Political Lean",
    farLeft: "Far Left",
    center: "Center",
    farRight: "Far Right",
    agentLog: "Agent Reasoning Log",
    runningAI: "Politican Analysis using AI...",
    orchestrating: "Orchestrating agent inference"
  },
  el: {
    navAnalyzer: "Ανάλυση",
    navLiveFeed: "Ζωντανή Ροή",
    heroSub: "Επικολλήστε ένα ελληνικό άρθρο ειδήσεων για άμεση ανάλυση AI.",
    analyzeBtn: "Ανάλυση",
    recentTitle: "Πρόσφατες Αναλύσεις",
    noRecent: "Δεν βρέθηκαν πρόσφατες αναλύσεις. Αναμονή για δεδομένα...",
    liveFeedTitle: "Ζωντανή Ροή Δεδομένων",
    waitingData: "Αναμονή για εισαγωγή εγγράφων στο MongoDB από το Spark...",
    prev: "Προηγ.",
    next: "Επόμ.",
    page: "Σελίδα",
    of: "από",
    analysisVerified: "Επιβεβαιωμένη Ανάλυση",
    source: "Πηγή:",
    noTags: "Δεν υπάρχουν ετικέτες",
    polLean: "Πολιτική Τάση",
    farLeft: "Αριστερά",
    center: "Κέντρο",
    farRight: "Δεξιά",
    agentLog: "Αιτιολόγηση AI",
    runningAI: "Εκτέλεση πολιτικής ανάλυσης με AI...",
    orchestrating: "Οργάνωση συμπερασμάτων"
  }
};

// ==========================================
// --- 2. BILINGUAL AI HELPERS ---
// ==========================================
// Formats dates safely from various formats (ISO, Timestamp, etc.)
const formatDate = (dateValue: any) => {
  if (!dateValue) return "";
  try {
    const d = new Date(dateValue);
    if (isNaN(d.getTime())) return dateValue; // Fallback if invalid
    return d.toLocaleString([], { 
      year: 'numeric', 
      month: 'numeric', 
      day: 'numeric', 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  } catch (e) {
    return dateValue;
  }
}

// Safely pulls the correct language from the AI's nested JSON
const getLocalizedText = (textData: any, lang: 'en' | 'el') => {
  if (!textData) return "";
  if (typeof textData === 'string') return textData; // Safety fallback for old database entries
  return textData[lang] || textData['el'] || textData['en'] || ""; 
}

const getLocalizedArray = (arrayData: any, lang: 'en' | 'el') => {
  if (!arrayData) return [];
  if (Array.isArray(arrayData)) return arrayData; // Safety fallback for old database entries
  return arrayData[lang] || arrayData['el'] || arrayData['en'] || [];
}

const getPolLeanBadgeColor = (lean: string) => {
  switch (lean) {
    case 'Left': return 'bg-red-100 text-red-700 border-red-200' 
    case 'Right': return 'bg-blue-100 text-blue-700 border-blue-200' 
    default: return 'bg-slate-100 text-slate-600 border-slate-200' 
  }
}

const getPolIndicatorClass = (score: number) => {
  if (score <= 40) return '[&>div]:bg-red-500' 
  if (score <= 60) return '[&>div]:bg-slate-400' 
  return '[&>div]:bg-blue-600' 
}

// ==========================================
// --- SHARED COMPONENT: THE MODAL ---
// ==========================================
const AnalysisModal = ({ isOpen, onClose, isAnalyzing, activeArticle }: any) => {
  const { lang } = useContext(LanguageContext);
  const text = t[lang];

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-slate-50 border-slate-200 text-slate-900 sm:max-w-3xl shadow-2xl overflow-hidden p-0 h-auto sm:h-[600px] flex flex-col">
        {isAnalyzing ? (
          <div className="flex-grow flex flex-col items-center justify-center space-y-6 bg-white min-h-[400px]">
            <div className="w-12 h-12 border-4 border-slate-100 border-t-red-600 rounded-full animate-spin"></div>
            <div className="space-y-2 text-center">
              <p className="text-slate-800 font-bold text-lg animate-pulse">{text.runningAI}</p>
              <p className="text-slate-500 font-medium">{text.orchestrating}</p>
            </div>
          </div>
        ) : (
          <>
            <div className="bg-white border-b border-slate-200 px-6 py-6 shrink-0">
              <DialogHeader>
                <div className="flex items-center gap-2 mb-2">
                  <ShieldCheck className="w-5 h-5 text-emerald-600" />
                  <span className="text-sm font-bold text-emerald-600 uppercase tracking-widest">{text.analysisVerified}</span>
                </div>
                {/* TRANSLATED TITLE */}
                <DialogTitle className="text-2xl font-extrabold text-slate-900 leading-tight">
                  {getLocalizedText(activeArticle?.title, lang)}
                </DialogTitle>
                <DialogDescription className="text-slate-500 mt-1 font-medium flex items-center gap-2">
                  {text.source}
                    <a href={activeArticle?.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline hover:text-blue-800 transition-colors">
                      {activeArticle?.source}
                    </a>
                    <span className="text-slate-300 mx-1">•</span>
                    <span className="text-slate-400 text-xs">{formatDate(activeArticle?.date)}</span>
                </DialogDescription>
                <DialogDescription className="text-slate-500 mt-1 font-medium flex items-center gap-2"> 
                  {/* TRANSLATED TAGS */}
                  {activeArticle?.tags && getLocalizedArray(activeArticle.tags, lang).length > 0 ? (
                    getLocalizedArray(activeArticle.tags, lang).map((tag: string, index: number) => (
                      <Badge key={index} variant="outline" className="text-slate-500 border-slate-200 bg-slate-50">{tag}</Badge>
                    ))
                  ) : (
                    <span className="text-slate-400">{text.noTags}</span>
                  )}
                </DialogDescription>
              </DialogHeader>
            </div>
            
            <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-50 flex-grow overflow-hidden">
              <div className="space-y-6">
                <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm h-full flex flex-col justify-center min-h-[160px]">
                  <div className="flex justify-between items-center mb-4">
                    <span className="font-bold text-slate-800 flex items-center gap-2">
                      <Scale className="w-4 h-4 text-slate-400"/> {text.polLean}
                    </span>
                    <Badge className={getPolLeanBadgeColor(activeArticle?.polLean)}>
                      {activeArticle?.polLean}
                    </Badge>
                  </div>
                  <div className="relative pt-1">
                    <Progress value={activeArticle?.polScore} className={`h-3 bg-slate-100 ${getPolIndicatorClass(activeArticle?.polScore)}`} />
                    <div className="flex mt-3 items-center justify-between text-[10px] text-slate-400 font-bold uppercase tracking-widest">
                      <span>{text.farLeft}</span><span>{text.center}</span><span>{text.farRight}</span>
                    </div>
                  </div>
                </div>
              </div>

              <Card className="bg-white border-slate-200 shadow-sm flex flex-col h-[350px] md:h-full">
                <CardHeader className="pb-3 border-b border-slate-100 bg-slate-50/50 shrink-0">
                  <CardTitle className="text-xs text-slate-500 font-bold uppercase tracking-widest flex items-center gap-2">
                    <FileText className="w-4 h-4" /> {text.agentLog}
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-4 text-slate-700 text-sm leading-relaxed font-medium overflow-y-auto flex-grow custom-scrollbar">
                  {/* TRANSLATED REASONING */}
                  {getLocalizedText(activeArticle?.reasoning, lang)}
                </CardContent>
              </Card>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

// ==========================================
// --- PAGE 1: THE MANUAL ANALYZER ---
// ==========================================
const AnalyzerPage = () => {
  const { lang } = useContext(LanguageContext);
  const text = t[lang];

  const [url, setUrl] = useState("")
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [activeArticle, setActiveArticle] = useState<any>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [recentArticles, setRecentArticles] = useState<any[]>([])

  useEffect(() => {
    const fetchRecent = async () => {
      try {
        const response = await fetch('/api/articles?limit=5')
        if (response.ok) {
          const data = await response.json()
          if (data && Array.isArray(data.articles)) {
            setRecentArticles(data.articles)
          }
        }
      } catch (error) { console.error("Failed to fetch recent articles:", error) }
    }
    fetchRecent()
    const interval = setInterval(fetchRecent, 10000)
    return () => clearInterval(interval)
  }, [])

  const handleAnalyze = async () => {
    if (!url) return
    setIsAnalyzing(true)
    setIsModalOpen(true)
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url })
      })
      if (response.ok) {
        setActiveArticle(await response.json())
      } else {
        setActiveArticle({ title: "Analysis Failed", reasoning: "Could not reach the AI agents." })
      }
    } catch (error) {
      setActiveArticle({ title: "System Error", reasoning: "Pipeline connection failed." })
    } finally {
      setIsAnalyzing(false)
      setUrl("") 
    }
  }

  return (
    <div className="max-w-3xl mx-auto mt-16 text-center space-y-6">
      <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight text-slate-900">
        Hellenic <span className="text-transparent bg-clip-text bg-gradient-to-r from-red-600 to-blue-600">Insight</span>
      </h1>
      <p className="text-slate-500 text-lg md:text-xl font-medium">
          {text.heroSub}
      </p>
      
      <div className="flex flex-col sm:flex-row gap-4 max-w-xl mx-auto mt-8">
        <div className="relative flex-grow">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 w-5 h-5" />
          <Input 
            placeholder="https://example.com/news-article..." 
            className="bg-white border-slate-300 text-slate-900 h-12 pl-10 text-md shadow-sm focus-visible:ring-red-500"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </div>
        <Button onClick={handleAnalyze} className="h-12 px-8 bg-slate-900 hover:bg-red-700 text-white font-semibold shadow-md transition-all flex items-center gap-2">
          <Activity className="w-4 h-4" /> {text.analyzeBtn}
        </Button>
      </div>

      <div className="mt-12">
        <div className="flex items-center gap-4 mb-6">
          <div className="h-px bg-slate-200 flex-grow"></div>
          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
            <Clock className="w-4 h-4" /> {text.recentTitle}
          </h3>
          <div className="h-px bg-slate-200 flex-grow"></div>
        </div>
        
        {recentArticles.length > 0 ? (
          <div className="flex flex-col gap-3 max-w-xl mx-auto text-left">
            {recentArticles.map((article, index) => (
              <div 
                key={index}
                onClick={() => { setActiveArticle(article); setIsModalOpen(true); }}
                className="group flex items-center justify-between p-4 bg-white/80 backdrop-blur-sm border border-slate-200 rounded-xl hover:border-red-300 hover:shadow-md transition-all cursor-pointer"
              >
                <div className="flex flex-col overflow-hidden mr-4">
                  {/* TRANSLATED RECENT TITLE */}
                  <span className="text-sm font-bold text-slate-800 truncate group-hover:text-red-600 transition-colors">
                    {getLocalizedText(article.title, lang)}
                  </span>
                  <span className="text-xs text-slate-500 mt-1 flex items-center gap-2">
                    <Badge variant="secondary" className="text-[10px] px-2 py-0 bg-white/50 text-slate-500 border-none">
                      {article.source}
                    </Badge>
                    {formatDate(article.date)}
                  </span>
                </div>
                <Badge className={`shrink-0 ${getPolLeanBadgeColor(article.polLean)}`}>
                  {article.polLean}
                </Badge>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-slate-400 text-sm bg-white/50 backdrop-blur-sm border border-slate-200 border-dashed rounded-xl max-w-xl mx-auto">
            {text.noRecent}
          </div>
        )}
      </div>

      <AnalysisModal isOpen={isModalOpen} onClose={setIsModalOpen} isAnalyzing={isAnalyzing} activeArticle={activeArticle} />
    </div>
  )
}

// ==========================================
// --- PAGE 2: THE LIVE FEED ---
// ==========================================
const LiveFeedPage = () => {
  const { lang } = useContext(LanguageContext);
  const text = t[lang];

  const [liveArticles, setLiveArticles] = useState<any[]>([])
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [activeArticle, setActiveArticle] = useState<any>(null)
  
  const [currentPage, setCurrentPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const ITEMS_PER_PAGE = 25 

  // --- FILTER BAR STATE & LOGIC ---
  const [selectedSource, setSelectedSource] = useState('All')
  const uniqueSources = ['All', ...new Set(liveArticles.map(article => article.source).filter(Boolean))];
  const filteredArticles = selectedSource === 'All' 
    ? liveArticles 
    : liveArticles.filter(article => article.source === selectedSource);

  useEffect(() => {
    const fetchLiveFeed = async () => {
      try {
        const response = await fetch(`/api/articles?page=${currentPage}&limit=${ITEMS_PER_PAGE}`)
        if (response.ok) {
          const data = await response.json()
          if (data && Array.isArray(data.articles)) {
            setLiveArticles(data.articles)
            setTotalPages(data.totalPages || 1)
          }
        }
      } catch (error) { console.error("Failed to fetch live feed:", error) }
    }
    fetchLiveFeed()
    const interval = setInterval(fetchLiveFeed, 30000)
    return () => clearInterval(interval)
  }, [currentPage])

  return (
    <div className="max-w-6xl mx-auto mt-10 pb-16">
      <div className="flex items-center gap-4 mb-8">
        <div className="h-px bg-slate-200 flex-grow opacity-50"></div>
        <h2 className="text-sm font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
          {text.liveFeedTitle}
        </h2>
        <div className="h-px bg-slate-200 flex-grow opacity-50"></div>
      </div>

      {/* --- NEW FILTER BAR UI --- */}
      {liveArticles.length > 0 && (
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {uniqueSources.map(source => (
            <button
              key={source}
              onClick={() => setSelectedSource(source)}
              className={`px-4 py-1.5 rounded-full text-xs font-bold transition-all duration-200 ${
                selectedSource === source 
                  ? 'bg-slate-800 text-white shadow-md' 
                  : 'bg-white text-slate-500 hover:bg-slate-100 border border-slate-200 hover:border-slate-300'
              }`}
            >
              {source}
            </button>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* --- USING FILTERED ARTICLES --- */}
        {filteredArticles.length > 0 ? (
          filteredArticles.map((article, index) => (
            <Card key={index} onClick={() => { setActiveArticle(article); setIsModalOpen(true); }} className="bg-white/80 border-slate-200 shadow-sm cursor-pointer hover:border-red-300 hover:bg-white hover:shadow-md transition-all group backdrop-blur-sm">
              <CardHeader>
                <div className="flex justify-between items-center mb-3">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-slate-500 border-slate-200 bg-white/50">{article.source}</Badge>
                    <span className="text-[10px] text-slate-400 font-medium">{formatDate(article.date)}</span>
                  </div>
                  <div className="w-12 h-1 bg-slate-200 rounded-full overflow-hidden opacity-60 group-hover:opacity-100 transition-opacity">
                    <div className={`h-full ${article.polScore <= 40 ? 'bg-red-500' : article.polScore <= 60 ? 'bg-slate-400' : 'bg-blue-600'}`} style={{ width: `${article.polScore}%` }} />
                  </div>
                </div>
                {/* TRANSLATED FEED TITLE */}
                <CardTitle className="text-slate-800 group-hover:text-red-600 transition-colors leading-snug text-base">
                  {getLocalizedText(article.title, lang)}
                </CardTitle>
              </CardHeader>
            </Card>
          ))
        ) : (
           <div className="col-span-full text-center py-20 text-slate-500 bg-white/50 backdrop-blur-sm border border-slate-200 border-dashed rounded-xl">
             {selectedSource !== 'All' ? 'No articles found for this source on this page.' : text.waitingData}
           </div>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex justify-center items-center gap-6 mt-12">
          <Button 
            variant="outline" 
            onClick={() => {
              setCurrentPage(prev => Math.max(prev - 1, 1));
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            disabled={currentPage === 1}
            className="bg-white/80 backdrop-blur-sm border-slate-200 text-slate-700 hover:bg-white disabled:opacity-50"
          >
            <ChevronLeft className="w-4 h-4 mr-1" /> {text.prev}
          </Button>
          
          <span className="text-sm font-bold text-slate-600 bg-white/50 px-4 py-2 rounded-full border border-slate-200 backdrop-blur-sm">
            {text.page} {currentPage} {text.of} {totalPages}
          </span>
          
          <Button 
            variant="outline" 
            onClick={() => {
              setCurrentPage(prev => Math.min(prev + 1, totalPages));
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            disabled={currentPage === totalPages}
            className="bg-white/80 backdrop-blur-sm border-slate-200 text-slate-700 hover:bg-white disabled:opacity-50"
          >
            {text.next} <ChevronRight className="w-4 h-4 ml-1" />
          </Button>
        </div>
      )}

      <AnalysisModal isOpen={isModalOpen} onClose={setIsModalOpen} isAnalyzing={false} activeArticle={activeArticle} />
    </div>
  )
}

// ==========================================
// --- THE NAVIGATION BAR ---
// ==========================================
const Navbar = () => {
  const location = useLocation();
  const { lang, setLang } = useContext(LanguageContext);
  const text = t[lang];
  
  return (
    <nav className="border-b border-slate-200 bg-white/80 backdrop-blur-md sticky top-0 z-50 w-full overflow-hidden">
      <div className="max-w-7xl mx-auto px-3 sm:px-6 h-16 flex items-center justify-between">
        
        {/* LEFT SIDE: Logo */}
        <div className="flex items-center gap-2 sm:gap-3 select-none shrink-0">
          <div className="flex items-center justify-center w-8 h-8 sm:w-10 sm:h-10 rounded-full border border-slate-300 bg-slate-50 shadow-sm">
            <Orbit className="w-4 h-4 sm:w-5 sm:h-5 text-slate-500" strokeWidth={1.5} />
          </div>
          <div className="flex flex-col justify-center">
            <span className="text-lg sm:text-xl font-extrabold tracking-tight text-slate-700 leading-none">
              BigO <span className="text-indigo-600/80">No</span>
            </span>
            <span className="text-[8px] sm:text-[9px] font-bold text-slate-400 uppercase tracking-[0.2em] mt-0.5">Powered By</span>
          </div>
        </div>

        {/* RIGHT SIDE: Buttons */}
        <div className="flex gap-1 sm:gap-2 items-center">
          {/* Language Toggle Button */}
          <Button 
            variant="outline" 
            onClick={() => setLang(lang === 'en' ? 'el' : 'en')}
            className="rounded-full px-2 sm:px-3 mr-1 sm:mr-4 border-slate-300 text-slate-600 hover:text-slate-900 bg-white/50 text-xs sm:text-sm h-9 sm:h-10"
          >
            <Globe className="w-3 h-3 sm:w-4 sm:h-4 sm:mr-2" />
            <span className="hidden sm:inline">{lang === 'en' ? 'EN' : 'GR'}</span>
            <span className="sm:hidden">{lang === 'en' ? 'EN' : 'GR'}</span>
          </Button>

          {/* Analyzer Button */}
          <Link to="/">
            <Button 
              variant={location.pathname === '/' ? 'default' : 'ghost'} 
              className={`px-3 sm:px-4 h-9 sm:h-10 ${location.pathname === '/' ? 'bg-slate-900 text-white' : 'text-slate-500'}`}
            >
              <Search className="w-4 h-4 sm:mr-2" /> 
              {/* This text hides on mobile, shows on larger screens */}
              <span className="hidden sm:inline">{text.navAnalyzer}</span> 
            </Button>
          </Link>

          {/* Live Feed Button */}
          <Link to="/feed">
            <Button 
              variant={location.pathname === '/feed' ? 'default' : 'ghost'} 
              className={`px-3 sm:px-4 h-9 sm:h-10 ${location.pathname === '/feed' ? 'bg-red-600 text-white hover:bg-red-700' : 'text-slate-500'}`}
            >
              <LayoutGrid className="w-4 h-4 sm:mr-2" /> 
              {/* This text hides on mobile, shows on larger screens */}
              <span className="hidden sm:inline">{text.navLiveFeed}</span>
            </Button>
          </Link>
        </div>

      </div>
    </nav>
  )
}

// ==========================================
// --- MAIN APP WRAPPER ---
// ==========================================
export default function App() {
  const [lang, setLang] = useState<Language>('en');

  return (
    <LanguageContext.Provider value={{ lang, setLang }}>
      <Router>
        <div className="min-h-screen text-slate-900 font-sans selection:bg-red-100 relative">
          <Navbar />
          <div className="p-8">
            <Routes>
              <Route path="/" element={<AnalyzerPage />} />
              <Route path="/feed" element={<LiveFeedPage />} />
            </Routes>
          </div>
        </div>
      </Router>
    </LanguageContext.Provider>
  )
}