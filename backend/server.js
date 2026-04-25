require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { MongoClient } = require('mongodb');

// --- NEW SCRAPER DEPENDENCIES ---
const cron = require('node-cron');
const { spawn } = require('child_process');
const path = require('path');
// --------------------------------

const app = express();
app.use(cors()); // Allows the frontend to talk to this backend

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

app.get('/api/articles', async (req, res) => {
    try {
        // Fetch articles and project out the _id field
        const articles = await collection.find({}, { projection: { _id: 0 } }).toArray();
        res.json(articles);
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Failed to fetch articles' });
    }
});

// ==========================================
// --- SEQUENTIAL SCRAPER PIPELINE LOGIC ---
// ==========================================

// Helper function to run a command and wait for it to finish
function runProcess(command, args, workingDirectory) {
    return new Promise((resolve, reject) => {
        const process = spawn(command, args, { cwd: workingDirectory });

        process.stdout.on('data', (data) => console.log(`[Scraper Output]: ${data.toString().trim()}`));
        process.stderr.on('data', (data) => console.error(`[Scraper Log/Error]: ${data.toString().trim()}`));

        process.on('close', (code) => {
            if (code === 0) {
                resolve();
            } else {
                reject(new Error(`Process exited with code ${code}`));
            }
        });
    });
}

// Schedule to run at minute 0 of every hour (e.g., 1:00, 2:00, 3:00)
cron.schedule('45 * * * *', async () => {
    console.log('Initiating hourly scraper pipeline...');
    
    // Path goes up one level from 'backend' into 'live_scraper'
    const scraperDir = path.join(__dirname, '../live_scraper');

    try {
        console.log('Step 1: Running Scrapy Spider...');
        // Runs: scrapy runspider live_news_spider.py -O raw_news.json
        await runProcess('scrapy', ['runspider', 'live_news_spider.py', '-O', 'raw_news.json'], scraperDir);

        console.log('Step 2: Running Cleaner...');
        // Runs: python live_cleaner.py
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
    console.log(`Server running on port ${PORT}`);
});