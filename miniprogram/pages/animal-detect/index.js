// miniprogram/pages/animal-detect/index.js
Page({
  data: {
    imagePath: "",
    fileID: "",
    loading: false,
    result: null
  },
  chooseImage() {
    wx.chooseImage({
      count: 1,
      sizeType: ["compressed"],
      sourceType: ["album", "camera"],
      success: (res) => {
        const path = res.tempFilePaths[0]
        this.setData({ imagePath: path, result: null, fileID: "" })
      }
    })
  },
  // 图片转Base64 (辅助方法)
  getImageBase64(tempFilePath) {
    return new Promise((resolve, reject) => {
      wx.getFileSystemManager().readFile({
        filePath: tempFilePath,
        encoding: 'base64',
        success: (res) => {
          resolve(`data:image/jpeg;base64,${res.data}`);
        },
        fail: reject
      });
    });
  },

  async detect() {
    if (!this.data.imagePath) return
    this.setData({ loading: true })
    try {
      // 方式一：直接传 Base64 给云函数 (完全遵循 test1.md)
      const base64Img = await this.getImageBase64(this.data.imagePath)
      
      const resp = await wx.cloud.callFunction({
        name: "animalPredict",
        config: {
          env: "cloud1-1g9tvkxmd8a7464e"
        },
        data: { 
          image: base64Img
        },
        timeout: 15000
      })
      
      const r = resp.result || {}
      if (r.success) {
         this.setData({ result: { label: r.class_name || "unknown", score: r.confidence || 0 } })
      } else {
         throw new Error(r.error || "识别失败")
      }

    } catch (e) {
      console.error(e)
      wx.showToast({ icon: "none", title: "识别失败: " + (e.message || "请检查服务") })
    } finally {
      this.setData({ loading: false })
    }
  },
  reset() {
    this.setData({ imagePath: "", fileID: "", result: null })
  }
})