const mongoose = require('mongoose');

const articlesSchema = new mongoose.Schema({
  source: {
    id: String,
    name: String
  },
  title: String,
  content: String,
  date: Date,
  bias: String,
});


module.exports = mongoose.model('articles', articlesSchema);