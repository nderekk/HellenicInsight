const article = require('../models/article_model');

exports.getArticles = async (req, res) => {
    try {
        const articles = await article.find();
        res.json(articles);
    } catch (err) {
        res.status(500).json({ message: err.message });
    }
};

exports.createArticle = async (req, res) => {
    const newArticle = new article({
        source: req.body.source,
        title: req.body.title,
        content: req.body.content,
        date: req.body.date,
        bias: req.body.bias
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