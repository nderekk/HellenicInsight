const mongoose = require('mongoose');

const articlesSchema = new mongoose.Schema({
  source: String,
  url: String,
  title: String,
  content: String,
  date: Date,
  ai_labels: {
    reasoning: String,
    primary_entities: [String],
    bias: Number
  }
});


module.exports = mongoose.model('articles', articlesSchema);