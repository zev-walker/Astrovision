# 🌌 AstroVision - Deployment Guide

## 📦 What You Have
- ✅ `app.py` - Complete application (single file)
- ✅ `requirements.txt` - All dependencies

## 🚀 Quick Deployment Steps

### Step 1: Create GitHub Repository

1. Go to **GitHub.com** and sign in
2. Click **"New"** to create a new repository
3. Name it: `astrovision`
4. Make it **Public**
5. ✅ Check "Add a README file"
6. Click **"Create repository"**

### Step 2: Upload Files to GitHub

**Option A: Using GitHub Website (Easiest)**

1. In your new repository, click **"Add file"** → **"Upload files"**
2. Drag and drop:
   - `app.py`
   - `requirements.txt`
3. Add commit message: "Initial commit - AstroVision app"
4. Click **"Commit changes"**

**Option B: Using VS Code (If you prefer)**

```bash
# In VS Code terminal (make sure you're in AstroVision folder)

# Initialize git
git init

# Add files
git add app.py requirements.txt

# Commit
git commit -m "Initial commit - AstroVision app"

# Link to your GitHub repo (replace YOUR-USERNAME)
git remote add origin https://github.com/YOUR-USERNAME/astrovision.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository:** `YOUR-USERNAME/astrovision`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**

**⏰ Wait 2-5 minutes for deployment...**

### Step 4: Your App is Live! 🎉

You'll get a URL like: `https://your-username-astrovision-app-xyz.streamlit.app`

Share this URL with anyone!

---

## 🧪 Testing Locally (Optional)

Before deploying, you can test locally:

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Opens in browser at `http://localhost:8501`

---

## 📝 Current App Features

### 🔭 Galaxy Classifier
- Upload galaxy images (JPG, PNG)
- AI classifies into 5 types:
  - Elliptical
  - Spiral
  - Barred Spiral
  - Irregular
  - Lenticular
- Shows confidence scores
- Visual probability charts

### 📚 Paper Analyzer
- Upload astronomy papers (PDF)
- Auto-generate summaries (3 lengths)
- Ask questions about paper
- Get instant answers
- Extract key insights

### 📊 Dashboard
- View usage statistics
- Classification trends
- Activity log

---

## ⚠️ Important Notes

### About the Current Model

**The app is using a DEMO model right now:**
- ✅ Works and demonstrates functionality
- ✅ Gives realistic-looking predictions
- ❌ Not trained on real galaxy data yet

**To get a REAL trained model later:**
1. Download actual galaxy dataset
2. Train model locally (I'll help you!)
3. Replace the model in the code
4. Redeploy

### NLP Models

The Paper Analyzer uses **pre-trained models** from Hugging Face:
- BART for summarization
- DistilBERT for Q&A
- These are **already trained** and work immediately!

---

## 🔧 Troubleshooting

### Problem: App won't deploy
**Check:**
- Files are in the root of the repository (not in a folder)
- `requirements.txt` is spelled correctly
- Repository is public

### Problem: Import errors
**Solution:** Streamlit Cloud will automatically install dependencies from `requirements.txt`. Wait for deployment to complete.

### Problem: NLP models too large
If you get memory errors, you can comment out the NLP section temporarily:
```python
# In app.py, comment out the Paper Analyzer page
# We'll optimize it later
```

### Problem: Slow first load
**This is normal!** First time loading takes 2-3 minutes as models download. After that, it's fast.

---

## 📈 Next Steps (After Deployment)

1. ✅ **Share your app** with friends/teachers
2. ✅ **Test all features** to make sure they work
3. ✅ **Get feedback** from users
4. 🎓 **Train your own model** (I'll help when ready!)
5. 🔄 **Replace demo model** with your trained one
6. ✨ **Add more features** (we can enhance it!)

---

## 🆘 Need Help?

Common issues and solutions:

| Issue | Solution |
|-------|----------|
| "No module named 'tensorflow'" | Check `requirements.txt` is in the repo |
| App crashes on image upload | Check image file size < 200MB |
| PDF analyzer not working | Ensure `PyPDF2` is in requirements |
| Slow predictions | Normal for first load, caches after |

---

## 🎯 When You're Ready to Train Your Own Model

**I'll help you with:**
1. Download proper galaxy dataset from Kaggle
2. Create training script
3. Train on your RTX 3050 (1-2 hours)
4. Export trained model
5. Upload to GitHub (handling large file)
6. Update app.py to use your model
7. Redeploy!

**Just let me know when you want to do this!** 🚀

---

## 📚 Resources

- **Streamlit Docs:** https://docs.streamlit.io
- **Galaxy Zoo Dataset:** https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge
- **Hugging Face Models:** https://huggingface.co/models
- **Your Live App:** (You'll get this after deployment!)

---

**Built with ❤️ for Astronomy and AI**

*Remember: This is your portfolio project. Be proud of it!* ⭐
