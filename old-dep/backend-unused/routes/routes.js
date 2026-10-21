const express = require('express');
const articleController = require('../controllers/article_controller');
const router = express.Router();

router.get('/articles', articleController.getArticles);
router.post('/articles', articleController.createArticle);
router.put('/articles/:id', articleController.modifyArticle);