import { useState, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
// NEW: Imported 'Orbit' for the clean vector logo symbol
import { ShieldCheck, BrainCircuit, Activity, Scale, Search, FileText, Orbit } from "lucide-react"
import VllmEye from "@/components/VllmEye"

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

export default function App() {
  const [url, setUrl] = useState("")
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [activeArticle, setActiveArticle] = useState<any>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  
  const [liveArticles, setLiveArticles] = useState<any[]>([])
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null)

  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null

    const fetchLiveFeed = async () => {
      try {
        const response = await fetch('/api/articles')
        if (response.ok) {
          const data = await response.json()
          setLiveArticles(data)
          setBackendOnline(true)
        } else {
          setBackendOnline(false)
          if (interval) clearInterval(interval)
        }
      } catch {
        setBackendOnline(false)
        if (interval) clearInterval(interval)
      }
    }

    fetchLiveFeed()
    interval = setInterval(fetchLiveFeed, 10000)
    return () => { if (interval) clearInterval(interval) }
  }, [])

  const handleAnalyze = async (existingArticle?: any) => {
    setIsAnalyzing(true)
    setIsModalOpen(true)
    
    if (existingArticle) {
      setActiveArticle(existingArticle)
      setIsAnalyzing(false)
      return
    }

    if (url) {
      try {
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url })
        })
        
        if (response.ok) {
          const aiResult = await response.json()
          setActiveArticle(aiResult)
        } else {
          setActiveArticle({ title: "Analysis Failed", reasoning: "Could not reach the Qwen AI agents." })
        }
      } catch (error) {
        console.error("Backend error:", error)
        setActiveArticle({ title: "System Error", reasoning: "Pipeline connection failed." })
      } finally {
        setIsAnalyzing(false)
      }
    }
  }

  return (
    <div className="min-h-screen text-slate-900 p-8 pb-20 md:pb-8 font-sans selection:bg-red-100 relative">

      {/* vLLM monitor — self-positioned: top-left on desktop, bottom strip on mobile */}
      <VllmEye />

      {/* PURE CSS BRANDING HEADER - Clean, professional, scalable */}
      <div className="absolute top-6 left-6 flex items-center gap-3 opacity-60 hover:opacity-100 transition-opacity select-none cursor-default">
        {/* Core Ring Symbol */}
        <div className="relative flex items-center justify-center w-10 h-10 rounded-full border border-slate-300 bg-white/50 shadow-sm backdrop-blur-sm">
          <Orbit className="w-5 h-5 text-slate-500" strokeWidth={1.5} />
        </div>
        
        {/* Typographic Logo */}
        <div className="flex flex-col justify-center">
          <span className="text-xl font-extrabold tracking-tight text-slate-700 leading-none">
            BigO <span className="text-indigo-600/80">No</span>
          </span>
          <span className="text-[9px] font-bold text-slate-400 uppercase tracking-[0.2em] mt-0.5">
            Powered By
          </span>
        </div>
      </div>

      {/* 1. HERO SECTION */}
      <div className="max-w-3xl mx-auto mt-20 text-center space-y-6">
        <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight text-slate-900">
          Hellenic <span className="text-transparent bg-clip-text bg-gradient-to-r from-red-600 to-blue-600">Insight</span>
        </h1>
        <p className="text-slate-500 text-lg md:text-xl font-medium">
           Paste an article below or browse the live pipeline feed.
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
          <Button 
            onClick={() => handleAnalyze()} 
            className="h-12 px-8 bg-slate-900 hover:bg-red-700 text-white font-semibold shadow-md transition-all flex items-center gap-2"
          >
            <Activity className="w-4 h-4" /> Analyze
          </Button>
        </div>
      </div>

      {/* 2. LIVE FEED GRID */}
      <div className="max-w-5xl mx-auto mt-32">
        <div className="flex items-center gap-4 mb-8">
          <div className="h-px bg-slate-200 flex-grow"></div>
          <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
            Live Article Feed
          </h2>
          <div className="h-px bg-slate-200 flex-grow"></div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {liveArticles.length > 0 ? (
            liveArticles.map((article, index) => (
              <Card 
                key={index} 
                className="bg-white/70 border-slate-200 shadow-sm cursor-pointer hover:border-red-300 hover:bg-white hover:shadow-md transition-all group backdrop-blur-sm"
                onClick={() => handleAnalyze(article)}
              >
                <CardHeader>
                  <div className="flex justify-between items-center mb-3">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-slate-500 border-slate-200 bg-slate-50">
                        {article.source}
                      </Badge>
                      <span className="text-[10px] text-slate-400 font-medium">
                        {article.date}
                      </span>
                    </div>
                    {/* Subtle Mini Meter */}
                    <div className="w-12 h-1 bg-slate-100 rounded-full overflow-hidden opacity-60 group-hover:opacity-100 transition-opacity">
                      <div 
                        className={`h-full ${article.polScore <= 40 ? 'bg-red-500' : article.polScore <= 60 ? 'bg-slate-400' : 'bg-blue-600'}`}
                        style={{ width: `${article.polScore}%` }}
                      />
                    </div>
                  </div>
                  <CardTitle className="text-slate-800 group-hover:text-red-600 transition-colors leading-snug">
                    {article.title}
                  </CardTitle>
                </CardHeader>
              </Card>
            ))
          ) : (
             <div className="col-span-3 text-center py-10 text-slate-500">
               Waiting for Spark pipeline to insert documents into MongoDB...
             </div>
          )}
        </div>
      </div>

      {/* 3. THE DASHBOARD MODAL */}
      <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
        <DialogContent className="bg-slate-50 border-slate-200 text-slate-900 sm:max-w-3xl shadow-2xl overflow-hidden p-0 h-[600px] flex flex-col">
          
          {isAnalyzing ? (
            <div className="flex-grow flex flex-col items-center justify-center space-y-6 bg-white">
              <div className="w-12 h-12 border-4 border-slate-100 border-t-red-600 rounded-full animate-spin"></div>
              <div className="space-y-2 text-center">
                <p className="text-slate-800 font-bold text-lg animate-pulse">Running Python AI Script...</p>
                <p className="text-slate-500 font-medium">Orchestrating agent inference</p>
              </div>
            </div>
          ) : (
            <>
              <div className="bg-white border-b border-slate-200 px-6 py-6">
                <DialogHeader>
                  <div className="flex items-center gap-2 mb-2">
                    <ShieldCheck className="w-5 h-5 text-emerald-600" />
                    <span className="text-sm font-bold text-emerald-600 uppercase tracking-widest">Analysis Verified</span>
                  </div>
                  <DialogTitle className="text-2xl font-extrabold text-slate-900 leading-tight">
                    {activeArticle?.title}
                  </DialogTitle>
                  <DialogDescription className="text-slate-500 mt-1 font-medium flex items-center gap-2">
                    Source:
                      <a 
                        href={activeArticle?.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:underline hover:text-blue-800 transition-colors"
                      >
                        {activeArticle?.source}
                      </a>
                      <span className="text-slate-300 mx-1">•</span>
                      <span className="text-slate-400 text-xs">{activeArticle?.date}</span>
                  </DialogDescription>
                  <DialogDescription className="text-slate-500 mt-1 font-medium flex items-center gap-2"> 
                    {activeArticle?.tags && activeArticle.tags.length > 0 ? (
                      activeArticle.tags.map((tag: string, index: number) => (
                        <Badge key={index} variant="outline" className="text-slate-500 border-slate-200 bg-slate-50">
                          {tag}
                        </Badge>
                      ))
                    ) : (
                      <span className="text-slate-400">No tags available</span>
                    )}
                  </DialogDescription>
                </DialogHeader>
              </div>
              
              <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-50 flex-grow">
                <div className="space-y-6 flex-grow">
                  <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm h-full flex flex-col justify-center">
                    <div className="flex justify-between items-center mb-4">
                      <span className="font-bold text-slate-800 flex items-center gap-2">
                        <Scale className="w-4 h-4 text-slate-400"/> Political Lean
                      </span>
                      <Badge className={getPolLeanBadgeColor(activeArticle?.polLean)}>
                        {activeArticle?.polLean}
                      </Badge>
                    </div>
                    <div className="relative pt-1">
                      <Progress 
                        value={activeArticle?.polScore} 
                        className={`h-3 bg-slate-100 ${getPolIndicatorClass(activeArticle?.polScore)}`}
                      />
                      <div className="flex mt-3 items-center justify-between text-[10px] text-slate-400 font-bold uppercase tracking-widest">
                        <span>Far Left</span>
                        <span>Center</span>
                        <span>Far Right</span>
                      </div>
                    </div>
                  </div>
                </div>

                <Card className="bg-white border-slate-200 shadow-sm h-full flex flex-col">
                  <CardHeader className="pb-3 border-b border-slate-100 bg-slate-50/50">
                    <CardTitle className="text-xs text-slate-500 font-bold uppercase tracking-widest flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      Agent Reasoning Log
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-4 text-slate-700 text-sm leading-relaxed font-medium flex-grow">
                    {activeArticle?.reasoning}
                  </CardContent>
                </Card>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}