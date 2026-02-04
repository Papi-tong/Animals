// pages/animal-detect/animal-detect.js
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
  async detect() {
    if (!this.data.imagePath) return
    this.setData({ loading: true })
    try {
      let fileID = this.data.fileID
      if (!fileID) {
        const cloudPath = `animals/${Date.now()}-${Math.floor(Math.random()*1e6)}.jpg`
        const upload = await wx.cloud.uploadFile({
          cloudPath,
          filePath: this.data.imagePath
        })
        fileID = upload.fileID
        this.setData({ fileID })
      }
      const resp = await wx.cloud.callFunction({
        name: "animalDetect",
        data: { fileID }
      })
      const r = resp.result || {}
      this.setData({ result: { label: r.label || "unknown", score: r.score || 0 } })
    } catch (e) {
      wx.showToast({ icon: "none", title: "识别失败" })
    } finally {
      this.setData({ loading: false })
    }
  },
  reset() {
    this.setData({ imagePath: "", fileID: "", result: null })
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad(options) {

  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady() {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow() {

  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide() {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload() {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh() {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom() {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage() {

  }
})