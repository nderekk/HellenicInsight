const article = require('../models/articles_model');

exports.getArticles = async (req, res) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 25;
        const skip = (page - 1) * limit;

        const total = await article.countDocuments();
        const articles = await article.find()
            .sort({ date: -1 }) // Sort by newest first
            .skip(skip)
            .limit(limit);

        res.json({
            articles,
            total,
            page,
            totalPages: Math.ceil(total / limit)
        });
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
};

exports.createArticle = async (req, res) => {
    const newArticle = new article({
        source: req.body.source,
        url: req.body.url,
        title: req.body.title,
        content: req.body.content,
        date: req.body.date,
        ai_labels: req.body.ai_labels
    });
    try {
        const savedArticle = await newArticle.save();
        res.status(201).json(savedArticle);
    } catch (err) {
        res.status(400).json({ message: err.message });
    }
};

exports.modifyArticle = async (req, res) => {
    try {
        const updatedArticle = await article.findByIdAndUpdate(req.params.id, req.body, { new: true });
        if (!updatedArticle) {
            return res.status(404).json({ message: 'Article not found' });
        }
        res.json(updatedArticle);
    } catch (err) {
        res.status(400).json({ message: err.message });
    }
};