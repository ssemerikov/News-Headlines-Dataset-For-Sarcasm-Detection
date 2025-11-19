# Deployment Guide - GitHub Pages

This guide explains how to deploy the sarcasm detection web interface to GitHub Pages.

## Prerequisites

- Git repository pushed to GitHub
- The `gh-pages/` folder committed to the main/master branch
- Repository Settings access

## Step-by-Step Deployment

### 1. Push to GitHub

First, ensure all changes are committed and pushed:

```bash
git status
git add .
git commit -m "Add web interface"
git push origin main  # or 'master' depending on your default branch
```

### 2. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** (gear icon in the top menu)
3. In the left sidebar, click **Pages**
4. Under **Source**, select:
   - **Branch**: `main` (or `master`)
   - **Folder**: `/gh-pages`
5. Click **Save**

### 3. Wait for Deployment

GitHub will build and deploy your site. This usually takes 1-2 minutes.

You can check the deployment status:
- Go to the **Actions** tab in your repository
- Look for the "pages build and deployment" workflow
- Wait for the green checkmark

### 4. Access Your Site

Once deployed, your site will be available at:

```
https://<your-username>.github.io/<repository-name>/
```

For example:
- Username: `johndoe`
- Repository: `News-Headlines-Dataset-For-Sarcasm-Detection`
- URL: `https://johndoe.github.io/News-Headlines-Dataset-For-Sarcasm-Detection/`

### 5. Update README

Add the live link to your README.md:

```markdown
## 🌐 Live Demo

**[Try the Sarcasm Detector](https://your-username.github.io/your-repo-name/)**
```

## Troubleshooting

### Site Not Loading

**Problem**: 404 error or blank page

**Solutions**:
1. Check that `/gh-pages` folder is selected (not root `/`)
2. Ensure `index.html` exists in `gh-pages/` folder
3. Wait 5-10 minutes for DNS propagation
4. Try clearing browser cache or using incognito mode

### Models Not Loading

**Problem**: "Failed to load model" error in browser console

**Solutions**:
1. Check browser console (F12) for specific error messages
2. Ensure model files are in `gh-pages/models/*/` directories
3. Verify file paths in `app.js` match actual structure
4. Check that binary `.bin` files were committed (not in `.gitignore`)
5. GitHub Pages has a 100 MB file size limit - our models (~5 MB each) are fine

### CORS Errors

**Problem**: Cross-origin errors when loading models

**Solution**: GitHub Pages automatically serves files with correct CORS headers. If you see CORS errors:
1. Make sure you're accessing via `https://` not `file://`
2. Try opening the live GitHub Pages URL instead of local file

## Local Testing

Before deploying, test locally:

```bash
cd gh-pages
python3 -m http.server 8000
# or
npx http-server -p 8000
```

Then open: `http://localhost:8000`

**Important**: Don't use `file:///` protocol - models won't load due to security restrictions.

## Custom Domain (Optional)

To use a custom domain:

1. Create a file named `CNAME` in `gh-pages/` folder
2. Add your domain: `sarcasm.yourdomain.com`
3. Configure DNS with your domain registrar:
   - Add CNAME record pointing to `<username>.github.io`
4. Push changes and wait for DNS propagation (up to 24 hours)

## Updating the Site

To update the deployed site:

1. Make changes to files in `gh-pages/`
2. Commit and push:
   ```bash
   git add gh-pages/
   git commit -m "Update web interface"
   git push origin main
   ```
3. GitHub Pages will automatically rebuild (1-2 minutes)

## Performance Tips

### Optimize Loading

1. **Model Loading**: All three models (~15 MB total) load on page load
   - Consider lazy loading models only when selected
   - Add service worker for offline caching

2. **Asset Optimization**:
   - CSS and JS are already minified via CDN (Bootstrap, TensorFlow.js)
   - Images: Not applicable (no images in this project)

3. **Caching**: GitHub Pages automatically sets cache headers

### Monitor Usage

GitHub Pages stats:
- **Bandwidth limit**: 100 GB/month (soft limit)
- **Build limit**: 10 builds/hour
- Our site: ~15 MB per visit, supports ~6,600 visits/month

## Security Notes

- All processing happens client-side
- No user data is sent to any server
- Models and code are publicly accessible (as intended)
- No API keys or secrets needed

## Next Steps

After deployment:

1. ✅ Test the live site thoroughly
2. ✅ Share the URL in your README
3. ✅ Consider adding analytics (Google Analytics, Plausible, etc.)
4. ✅ Monitor GitHub Issues for user feedback
5. ✅ Update models periodically as you retrain

## Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [This Repository's gh-pages README](./gh-pages/README.md)
