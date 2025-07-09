# 文档部署指南

## 本地预览

1. 安装docsify-cli:
```bash
npm install -g docsify-cli
```

2. 启动本地服务器:
```bash
docsify serve docs
```
访问 http://localhost:3000

## GitHub Pages部署

1. 创建gh-pages分支:
```bash
git checkout --orphan gh-pages
git rm -rf .
```

2. 复制文档文件:
```bash
cp -r docs/* .
```

3. 提交并推送:
```bash
git add .
git commit -m "部署文档到GitHub Pages"
git push origin gh-pages
```

4. 启用GitHub Pages:
- 进入仓库Settings > Pages
- 选择`gh-pages`分支和`/ (root)`目录
- 点击Save

5. 访问文档:
- 文档将发布在:  
  `https://<用户名>.github.io/<仓库名>`

## 高级部署选项

### Docker部署
```dockerfile
FROM node:alpine
RUN npm install -g docsify-cli
COPY docs /docs
WORKDIR /docs
EXPOSE 3000
CMD ["docsify", "serve", "--port", "3000"]
```

### ReadTheDocs部署
1. 注册ReadTheDocs账号
2. 导入项目仓库
3. 配置为静态HTML站点
4. 设置构建目录为`docs`