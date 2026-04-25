import { useState, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ShieldCheck, BrainCircuit, Activity, Scale, Search, FileText } from "lucide-react"

export default function App() {
  const [url, setUrl] = useState("")
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [activeArticle, setActiveArticle] = useState<any>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  
  // NEW: State to hold your live MongoDB feed
  const [liveArticles, setLiveArticles] = useState<any[]>([])

  // NEW: Fetch the live feed when the page loads
  useEffect(() => {
    const fetchLiveFeed = async () => {
      try {
        const response = await fetch('/api/articles')
        if (response.ok) {
          const data = await response.json()
          setLiveArticles(data)
        }
      } catch (error) {
        console.error("Failed to fetch live feed:", error)
      }
    }

    fetchLiveFeed()
    // Refresh the feed every 10 seconds automatically!
    const interval = setInterval(fetchLiveFeed, 10000)
    return () => clearInterval(interval)
  }, [])

  // UPGRADED: Send URL to Python Script
  const handleAnalyze = async (existingArticle?: any) => {
    setIsAnalyzing(true)
    setIsModalOpen(true)
    
    // If they clicked a card from the feed, show that data instantly
    if (existingArticle) {
      setActiveArticle(existingArticle)
      setIsAnalyzing(false)
      return
    }

    // If they pasted a URL, send it to Python!
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
    <div className="min-h-screen bg-slate-50 text-slate-900 p-8 font-sans selection:bg-indigo-100">
      
      {/* 1. HERO SECTION */}
      <div className="max-w-3xl mx-auto mt-20 text-center space-y-6">
        <Badge variant="outline" className="border-indigo-200 text-indigo-700 bg-indigo-50 mb-4 py-1.5 px-4 shadow-sm flex items-center w-fit mx-auto gap-2">
          <BrainCircuit className="w-4 h-4" />
          Qwen LoRA Expert Agents Online
        </Badge>
        <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight text-slate-900">
          Uncover the Hidden <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-cyan-500">Bias</span>
        </h1>
        <p className="text-slate-500 text-lg md:text-xl font-medium">
          Paste an article below or browse the live pipeline feed.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 max-w-xl mx-auto mt-8">
          <div className="relative flex-grow">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 w-5 h-5" />
            <Input 
              placeholder="https://example.com/news-article..." 
              className="bg-white border-slate-300 text-slate-900 h-12 pl-10 text-md shadow-sm focus-visible:ring-indigo-500"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
          </div>
          <Button 
            onClick={() => handleAnalyze()} 
            className="h-12 px-8 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold shadow-md transition-all flex items-center gap-2"
          >
            <Activity className="w-4 h-4" /> Analyze
          </Button>
        </div>
      </div>

      {/* 2. LIVE FEED GRID (Now maps over liveArticles!) */}
      <div className="max-w-5xl mx-auto mt-32">
        <div className="flex items-center gap-4 mb-8">
          <div className="h-px bg-slate-200 flex-grow"></div>
          <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            Live Database Feed
          </h2>
          <div className="h-px bg-slate-200 flex-grow"></div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {liveArticles.length > 0 ? (
            liveArticles.map((article, index) => (
              <Card 
                key={index} 
                className="bg-white border-slate-200 shadow-sm cursor-pointer hover:border-indigo-300 hover:shadow-md transition-all group"
                onClick={() => handleAnalyze(article)}
              >
                <CardHeader>
                  <Badge variant="outline" className="w-fit mb-3 text-slate-500 border-slate-200 bg-slate-50">
                    {article.source}
                  </Badge>
                  <CardTitle className="text-slate-800 group-hover:text-indigo-600 transition-colors leading-snug">
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
        <DialogContent className="bg-slate-50 border-slate-200 text-slate-900 sm:max-w-3xl shadow-2xl overflow-hidden p-0">
          
          {isAnalyzing ? (
            <div className="py-32 flex flex-col items-center justify-center space-y-6 bg-white">
              <div className="w-12 h-12 border-4 border-indigo-100 border-t-indigo-600 rounded-full animate-spin"></div>
              <div className="space-y-2 text-center">
                <p className="text-slate-800 font-bold text-lg animate-pulse">Running Python AI Script...</p>
                <p className="text-slate-500 font-medium">Please wait for Qwen inference to complete</p>
              </div>
            </div>
          ) : (
            <>
              {/* Modal Header */}
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
                    Source: <span className="text-slate-700">{activeArticle?.source}</span>
                  </DialogDescription>
                </DialogHeader>
              </div>
              
              {/* Modal Body - Two Column Grid */}
              <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-50">
                <div className="space-y-6">
                  {/* Political Meter */}
                  <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
                    <div className="flex justify-between items-center mb-4">
                      <span className="font-bold text-slate-800 flex items-center gap-2">
                        <Scale className="w-4 h-4 text-slate-400"/> Political Lean
                      </span>
                      <Badge className={
                        activeArticle?.polLean === 'Left' ? 'bg-blue-100 text-blue-700 border-blue-200' : 
                        activeArticle?.polLean === 'Right' ? 'bg-red-100 text-red-700 border-red-200' : 
                        'bg-purple-100 text-purple-700 border-purple-200'
                      }>
                        {activeArticle?.polLean}
                      </Badge>
                    </div>
                    <div className="relative pt-1">
                      <Progress value={activeArticle?.polScore} className="h-2.5 bg-slate-100 [&>div]:bg-slate-800" />
                      <div className="flex mt-2 items-center justify-between text-[10px] text-slate-400 font-bold uppercase tracking-widest">
                        <span>Far Left</span>
                        <span>Center</span>
                        <span>Far Right</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Reasoning Log */}
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