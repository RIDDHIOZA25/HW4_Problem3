# GitHub & Streamlit Deployment Commands

## Step 1: Initialize Git Repository

```bash
# Initialize git repository
git init

# Add all files (except those in .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: ReAct Agent with Streamlit UI"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `react-agent-streamlit`)
3. **DO NOT** initialize with README, .gitignore, or license (we already have these)

## Step 3: Connect and Push to GitHub

```bash
# Add your GitHub repository as remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Example:**
```bash
git remote add origin https://github.com/riddh/react-agent-streamlit.git
git branch -M main
git push -u origin main
```

## Step 4: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch (`main`)
5. Set the main file path: `streamlit_app.py`
6. Click "Deploy"

## Important Notes:

- Make sure your `.env` file is **NOT** committed (it's in .gitignore)
- For Streamlit Cloud, you'll need to add your API keys as secrets:
  - Go to your app settings → "Secrets"
  - Add:
    ```
    GOOGLE_API_KEY=your_actual_google_api_key
    TAVILY_API_KEY=your_actual_tavily_api_key
    ```

## Future Updates:

```bash
# After making changes
git add .
git commit -m "Your commit message"
git push
```

Streamlit will automatically redeploy when you push to the main branch!

