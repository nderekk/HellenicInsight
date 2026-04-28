require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { MongoClient } = require('mongodb');

// --- NEW SCRAPER DEPENDENCIES ---
const cron = require('node-cron');
// Notice I added 'exec' here so we can run the Fast Lane script!
const { spawn, exec } = require('child_process'); 
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
        const rawArticles = await collection.find({}, { projection: { _id: 0 } }).toArray();
        
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
                tags: doc.ai_labels?.primary_entities || [],
                reasoning: doc.ai_labels?.reasoning || "Analysis pending...",
                polLean: calculatedLean,
                polScore: percentageScore
            };
        });

        res.json(formattedArticles);
        
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Failed to fetch articles' });
    }
});


// ==========================================
// --- ROUTE 2: THE VIP FAST LANE (ANALYZE) -
// ==========================================
app.post('/api/analyze', (req, res) => {
    const targetUrl = req.body.url;

    if (!targetUrl) {
        return res.status(400).json({ error: "URL is required" });
    }

    console.log(`[Fast Lane] React requested analysis for: ${targetUrl}`);
    console.log(`[Fast Lane] Booting up Crawl4AI & Qwen...`);
    
    exec(`python qwen_analyzer.py "${targetUrl}"`, { cwd: __dirname }, (error, stdout, stderr) => {
        if (error) {
            console.error(`[Fast Lane Error]: ${error.message}`);
            return res.status(500).json({ error: 'AI processing failed' });
        }
        
        try {
            const jsonStartIndex = stdout.indexOf('{');
            if (jsonStartIndex === -1) throw new Error("No JSON found in Python output");
            
            const cleanJsonString = stdout.substring(jsonStartIndex);
            const aiResult = JSON.parse(cleanJsonString);
            
            console.log(`[Fast Lane] Success! Sending data back to React.`);
            // Send the perfectly formatted data back to your dashboard!
            res.json(aiResult);

        } catch (parseError) {
            console.error('[Fast Lane Error] Failed to parse Python output. Python printed:', stdout);
            res.status(500).json({ error: 'Invalid data format returned from AI' });
        }
    });
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

cron.schedule('45 * * * *', async () => {
    console.log('Initiating hourly scraper pipeline...');
    const scraperDir = path.join(__dirname, '../live_scraper');

    try {
        console.log('Step 1: Running Scrapy Spider...');
        await runProcess('scrapy', ['runspider', 'live_news_spider.py', '-O', 'raw_news.json'], scraperDir);

        console.log('Step 2: Running Cleaner...');
        await runProcess('python', ['live_cleaner.py'], scraperDir);

        console.log('Pipeline finished! kafka_feed.jsonl is updated and ready.');

        console.log('Step 3 : Sending to kafka');
        await runProcess('python',['producer.py'],__dirname);
        
    } catch (error) {
        console.error('Pipeline failed during execution:', error);
    }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server running on port http://localhost:${PORT}`);
});