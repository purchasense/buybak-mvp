// action - customization reducer
export const SET_MENU = '@customization/SET_MENU';
export const MENU_TOGGLE = '@customization/MENU_TOGGLE';
export const MENU_OPEN = '@customization/MENU_OPEN';
export const SET_FONT_FAMILY = '@customization/SET_FONT_FAMILY';
export const SET_BORDER_RADIUS = '@customization/SET_BORDER_RADIUS';

export const SET_MODAL_QRCODE_STATUS = "@buybak/SET_MODAL_QRCODE_STATUS";
export const SET_MODAL_QRCODE_LOADING_STATUS = "@buybak/SET_MODAL_QRCODE_LOADING_STATUS";
export const SET_MODAL_QRCODE_LOADING_EXEC_STATUS = "@buybak/SET_MODAL_QRCODE_LOADING_EXEC_STATUS";
export const SET_MODAL_QRCODE_SCAN = "@buybak/SET_MODAL_QRCODE_SCAN";
export const SET_MODAL_QRCODE_SELL = "@buybak/SET_MODAL_QRCODE_SELL";
export const SET_BUYBAK_RESET_ALERT_COUNT = "@buybak/SET_BUYBAK_RESET_ALERT_COUNT";
export const SET_STOCK_QUOTES = "@buybak/SET_STOCK_QUOTES";
export const SET_MODAL_MOBILE_LOGIN_STATUS = "@buybak/SET_MODAL_MOBILE_LOGIN_STATUS";
export const SET_MODAL_MOBILE_LOGIN_NAME = "@buybak/SET_MODAL_MOBILE_LOGIN_NAME";
export const SET_BUYBAK_STATISTICS = "@buybak/SET_BUYBAK_STATISTICS";
export const SET_BUYBAK_MOBILE_MESSAGE = "@buybak/SET_BUYBAK_MOBILE_MESSAGE";
export const GET_BUYBAK_MOBILE_MESSAGES = "@buybak/GET_BUYBAK_MOBILE_MESSAGES";
export const SET_BUYBAK_PREDICTIONS = "@buybak/SET_BUYBAK_PREDICTIONS";
export const SET_BUYBAK_FORECASTORS = "@buybak/SET_BUYBAK_FORECASTORS";

export function setModalQRCodeStatus(isOpen, store_id) {
  return { type: SET_MODAL_QRCODE_STATUS, isOpen, store_id };
}

export function setModalQRCodeLoadingStatus(isLoadingOpen, store_id) {
  return { type: SET_MODAL_QRCODE_LOADING_STATUS, isLoadingOpen, store_id };
}

export function setModalQRCodeLoadingExecutionStatus(executionStatus, store_id) {
  return { type: SET_MODAL_QRCODE_LOADING_EXEC_STATUS, executionStatus, store_id };
}

export function setModalQRCodeScan(store_id, fsop, stock_price) {
  return { type: SET_MODAL_QRCODE_SCAN, store_id, fsop, stock_price };
}

export function setModalQRCodeSell(store_id, fsop, stock_price) {
  return { type: SET_MODAL_QRCODE_SELL, store_id, fsop, stock_price };
}

export function setBuybakResetAlertCount(store_id, fsop) {
  return { type: SET_BUYBAK_RESET_ALERT_COUNT};
}

export function setStockQuotes(store_id, quantity, stock_price) {
  return { type: SET_STOCK_QUOTES, store_id, quantity, stock_price};
}

export function setModalMobileLoginName(username) {
  return { type: SET_MODAL_MOBILE_LOGIN_NAME, username};
}

export function setModalMobileLoginStatus(isOpen) {
  return { type: SET_MODAL_MOBILE_LOGIN_STATUS, isOpen};
}

export function setBuybakStatistics(store_id, total_trans, total_value) {
  return { type: SET_BUYBAK_STATISTICS, store_id, total_trans, total_value};
}

export function setBuybakMobileMessage(id, user, message) {
  return { type: SET_BUYBAK_MOBILE_MESSAGE, id, user, message};
}

export function getBuybakMobileMessages() {
  return { type: GET_BUYBAK_MOBILE_MESSAGES};
}

export function setBuybakPredictions(values) {
  return { type: SET_BUYBAK_PREDICTIONS, values};
}

export function setBuybakForecastors(values) {
  return { type: SET_BUYBAK_FORECASTORS, values};
}

