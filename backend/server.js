require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { MongoClient } = require('mongodb');

// --- NEW SCRAPER DEPENDENCIES ---
const cron = require('node-cron');
// Notice I added 'exec' here so we can run the Fast Lane script!
const { spawn } = require('child_process');
const path = require('path');
const { url } = require('inspector');
// --------------------------------

const app = express();

// --- CRITICAL FIX: Added express.json() so Node can read React's POST data ---
app.use(cors()); 
app.use(express.json()); 

const uri = process.env.MONGO_URI;
const dbName = process.env.DB_NAME;
const collectionName = process.env.COLLECTION_NAME;

let collection;

// Initialize MongoDB Connection
MongoClient.connect(uri)
    .then(client => {
        console.log('Connected to MongoDB');
        const db = client.db(dbName);
        collection = db.collection(collectionName);
    })
    .catch(error => {
        console.error('Failed to connect to MongoDB', error);
    });


// ==========================================
// --- ROUTE 1: THE LIVE FEED (BACKGROUND) --
// ==========================================
app.get('/api/articles', async (req, res) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 25;
        const skip = (page - 1) * limit;

        const total = await collection.countDocuments();
        const rawArticles = await collection.find({}, { projection: { _id: 0 } })
            .sort({ date: -1 })
            .skip(skip)
            .limit(limit)
            .toArray();

        const formattedArticles = rawArticles.map(doc => {
            const rawBias = doc.ai_labels?.bias || 0.5;
            const percentageScore = rawBias * 100;
            
            let calculatedLean = "Center";
            if (rawBias < 0.4) calculatedLean = "Left";
            if (rawBias > 0.6) calculatedLean = "Right";

            return {
                title: doc.title || "No Title",
                source: doc.source || "Unknown Source",
                url: doc.url || "#",
                date: doc.date || "",
                tags: doc.ai_labels?.primary_entities || [],
                reasoning: doc.ai_labels?.reasoning || "Analysis pending...",
                polLean: calculatedLean,
                polScore: percentageScore
            };
        });

        res.json({
            articles: formattedArticles,
            total,
            page,
            totalPages: Math.ceil(total / limit)
        });
        
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Failed to fetch articles' });
    }
});


// ==========================================
// --- ROUTE 2: THE VIP FAST LANE (ANALYZE) -
// ==========================================
app.post('/api/analyze', async (req, res) => {
    const targetUrl = req.body.url;

    if (!targetUrl) {
        return res.status(400).json({ error: "URL is required" });
    }

    console.log(`[Fast Lane] Analyzing: ${targetUrl}`);

    try {
        const response = await fetch('http://127.0.0.1:5002/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: targetUrl }),
        });

        if (!response.ok) {
            throw new Error(`Analyzer returned status ${response.status}`);
        }

        const aiResult = await response.json();
        console.log(`[Fast Lane] Success!`);
        res.json(aiResult);

    } catch (error) {
        console.error(`[Fast Lane Error]: ${error.message}`);
        res.status(500).json({ error: 'AI processing failed. Is qwen_analyzer.py running?' });
    }
});


// ==========================================
// --- SEQUENTIAL SCRAPER PIPELINE LOGIC ---
// ==========================================

function runProcess(command, args, workingDirectory) {
    return new Promise((resolve, reject) => {
        const process = spawn(command, args, { cwd: workingDirectory });
        process.stdout.on('data', (data) => console.log(`[Scraper Output]: ${data.toString().trim()}`));
        process.stderr.on('data', (data) => console.error(`[Scraper Log/Error]: ${data.toString().trim()}`));
        process.on('close', (code) => {
            if (code === 0) resolve();
            else reject(new Error(`Process exited with code ${code}`));
        });
    });
}

// cron.schedule('00 * * * *', async () => {
//     console.log('Initiating hourly scraper pipeline...');
//     const scraperDir = path.join(__dirname, '../live_scraper');

//     try {
//         console.log('Step 1: Running Scrapy Spider...');
//         await runProcess('scrapy', ['runspider', 'live_news_spider.py', '-O', 'raw_news.jsonl'], scraperDir);

//         console.log('Step 2: Running Cleaner...');
//         await runProcess('python', ['live_cleaner.py'], scraperDir);

//         console.log('Pipeline finished! kafka_feed.jsonl is updated and ready.');

//         console.log('Step 3 : Sending to kafka');
//         await runProcess('python',['producer.py'],__dirname);
        
//     } catch (error) {
//         console.error('Pipeline failed during execution:', error);
//     }
// });

const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
    console.log(`Server running on port http://localhost:${PORT}`);
});