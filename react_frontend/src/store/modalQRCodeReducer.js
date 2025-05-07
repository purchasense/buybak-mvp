import * as actionTypes from "./actions";
import * as I from "immutable";

export class Customer {

    constructor( name, cid, email, phone, type, address)
    {
        this.name = name;
        this.cid = cid;
        this.email = email;
        this.phone = phone;
        this.type = type;
        this.address = address;
    }
}

export class Retailer {

    constructor( store_name, store_id, store_number, city, province, zip, store_logo)
    {
        this.store_name = store_name;
        this.store_id = store_id;
        this.store_number = store_number;
        this.city = city;
        this.province = province;
        this.zip = zip;
        this.store_logo = store_logo;
    }
}

export class CustomerRetailFSOP {
    
    constructor( customer: Customer, retailer: Retailer, stock_price: Number, fsop: Number)
    {
        this.customer = customer;
        this.retailer = retailer;
        this.stock_price = stock_price;
        this.fsop = fsop;
    }
}

export class BuybakStatistics {
    
    constructor( store_id, total_value: Number, total_trans: Number)
    {
        this.store_id = store_id;
        this.total_value = total_value;
        this.total_trans = total_trans;
    }
}

export class MobileChatMessage {
    
    constructor( id: Number, user: String, msg: String)
    {
        this.id = id;
        this.user = user;
        this.msg = msg;
    }
}

export class WineSelection {

    constructor( id: String, image: String, place: String, name: String, title: String, notes: String)
    {
        this.id = id;
        this.image = image;
        this.place = place;
        this.name = name;
        this.title = title;
        this.notes = notes;
    }
}

const map_store_to_images = new I.Map({
    'AAPL':   '/images/AppleLogo.png',
    'AMZN':   '/images/AmazonLogo.png',
    'BP':   '/images/BPLogo.png',
    'CVS':   '/images/CVSLogo.png',
    'LLY':   '/images/LillyLogo.png',
    'NVDA':   '/images/NvidiaLogo.png',
    'SHEL':   '/images/ShellLogo.png',
    'TSLA':   '/images/TeslaLogo.png',
    'WMT':   '/images/WalmartLogo.png',
    'XOM':   '/images/ExxonLogo.png',
    'HD':   '/images/homedepot.png',
    'TGT':   '/images/target.png',
    'CMG':   '/images/chipotle.png',
    'SBUX':   '/images/starbucks.png',
    'ACE':   '/images/ace.png',
    'LOW':  '/images/lowes.png',
    'COST': '/images/costco.png',
    'WBA':   '/images/walgreens.png',
});

const initialState = {
  isOpen: false,
  isLoadingOpen: false,
  executionStatus: "Starting",
  alertCount: 0,
  is_login_open: true,
  login_username: undefined,
  total_fsop: new Number(1099.00),
  total_curr: new Number(1099.00),
  cdata: [0, 100, 175, 333, 500, 555, 689, 876, 989, 1000, 1103],
  predictions: [],
  forecastors: [],
  map_store_to_quotes: new I.Map({
    'AAPL':   new CustomerRetailFSOP( 
                new Customer('', '', '', '', '', ''), 
                new Retailer('Apple', 'AAPL', '1', 'Naperville', 'IL', '60563', '/images/AppleLogo.png'), 1722800, 100), 
    'AMZN':   new CustomerRetailFSOP( 
                new Customer('', '', '', '', '', ''), 
                new Retailer('Amazon', 'AMZN', '1', 'Naperville', 'IL', '60563', '/images/AmazonLogo.png'), 1788700, 100), 
    'BP':   new CustomerRetailFSOP( 
                new Customer('', '', '', '', '', ''), 
                new Retailer('BP PLC', 'BP', '1', 'Naperville', 'IL', '60563', '/images/BPLogo.png'), 378000, 100), 
    'CVS':   new CustomerRetailFSOP( 
                new Customer('', '', '', '', '', ''), 
                new Retailer('CVS Health', 'CVS', '1', 'Naperville', 'IL', '60563', '/images/CVSLogo.png'), 784800, 100), 
    'LLY':   new CustomerRetailFSOP( 
                new Customer('', '', '', '', '', ''), 
                new Retailer('Eli Lilly and Co', 'LLY', '1', 'Naperville', 'IL', '60563', '/images/LillyLogo.png'), 7706100, 100), 
    'NVDA':   new CustomerRetailFSOP( 
                new Customer('', '', '', '', '', ''), 
                new Retailer('NVIDIA', 'NVDA', '1', 'Naperville', 'IL', '60563', '/images/NvidiaLogo.png'), 9428900, 100), 
    'SHEL':   new CustomerRetailFSOP( 
                new Customer('', '', '', '', '', ''), 
                new Retailer('Shell PLC', 'SHEL', '1', 'Naperville', 'IL', '60563', '/images/ShellLogo.png'), 669200, 100), 
    'TSLA':   new CustomerRetailFSOP( 
                new Customer('', '', '', '', '', ''), 
                new Retailer('Tesla', 'TSLA', '1', 'Naperville', 'IL', '60563', '/images/TeslaLogo.png'), 1708301, 100), 
    'WMT':   new CustomerRetailFSOP( 
                new Customer('', '', '', '', '', ''), 
                new Retailer('Walmart', 'WMT', '1', 'Naperville', 'IL', '60563', '/images/WalmartLogo.png'), 608700, 100), 
    'XOM':   new CustomerRetailFSOP( 
                new Customer('', '', '', '', '', ''), 
                new Retailer('Exxon Mobil', 'XOM', '1', 'Naperville', 'IL', '60563', '/images/ExxonLogo.png'), 1134900, 100), 
    'HD':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('HomeDepot', 'HD', '171', 'Naperville', 'IL', '60563', "/images/homedepot.png"), 3631500, 1100),
    'TGT':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Target', 'TGT', '122', 'Naperville', 'IL', '60563', "/images/target.png"), 1463500, 2122),
    'CMG':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Chipotle', 'CMG', '183', 'Naperville', 'IL', '60563', "/images/chipotle.png"), 26382500, 1199),
    'SBUX':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Starbucks', 'SBUX', '115', 'Naperville', 'IL', '60563', "/images/starbucks.png"), 971500, 280),
    'ACE':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Ace Hard.', 'ACE', '116', 'Naperville', 'IL', '60563', "/images/ace.png"), 452500, 1235),
    'LOW':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Lowes', 'LOW', '117', 'Naperville', 'IL', '60563', "/images/lowes.png"), 2222600, 543),
    'COST':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Costco', 'COST', '118', 'Naperville', 'IL', '60563', "/images/costco.png"), 7234000, 980),
    'WBA':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Walgreens', 'WBA', '164', 'Naperville', 'IL', '60563', "/images/walgreens.png"), 222400, 2211),
  }),
  map_store_to_fsop: new I.Map({
    'HD':   new CustomerRetailFSOP( 
                    new Customer('Sameer', '0x1010', 'sameer@buybak.xyz', '630-696-7660', 'construction', 'Naperville, IL'), 
                    new Retailer('HomeDepot', 'HD', '171', 'Naperville', 'IL', '60563', "/images/homedepot.png"), 3631500, 285),
    }),
  map_store_to_mobile_messages: new I.Map({
    1745754093973:   new MobileChatMessage( 1745754093973, 'sameer', '```HTML <div> <p> Hi ChatGPT</p> </div> ```'),
    1745754094974:   new MobileChatMessage( 1745754094974, 'GPT', '```HTML <table> <tr> <th>Departure</th> <th>Total Time</th> <th>Airport Codes</th> <th>Price</th> </tr> <tr> <td>7:20 pm</td> <td>20h 25m</td> <td>ORD-GOX</td> <td>1917.0</td> </tr> <tr> <td>4:10 am</td> <td>21h 50m</td> <td>GOX-ORD</td> <td>1430.0</td> </tr> <tr> <td>7:20 pm</td> <td>20h 25m</td> <td>ORD-GOX</td> <td>1921.0</td> </tr> <tr> <td>4:10 am</td> <td>21h 50m</td> <td>GOX-ORD</td> <td>1919.0</td> </tr> </table> ```'),
    1745754095975:   new MobileChatMessage( 1745754095975, 'sameer', '```HTML <div> <p> Hi ChatGPT</p> </div> ```'),
    1745754096976:   new MobileChatMessage( 1745754096976, 'GPT', ' <table> <tr> <th>Date</th> <th>Chosen High</th> </tr> <tr> <td>Jan 17 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 18 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 19 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 20 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 21 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 22 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 23 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 24 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 25 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 26 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 27 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 28 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 29 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 30 2025</td> <td>775.28</td> </tr> <tr> <td>Jan 31 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 1 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 2 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 3 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 4 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 5 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 6 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 7 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 8 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 9 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 10 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 11 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 12 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 13 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 14 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 15 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 16 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 17 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 18 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 19 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 20 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 21 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 22 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 23 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 24 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 25 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 26 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 27 2025</td> <td>775.28</td> </tr> <tr> <td>Feb 28 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 1 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 2 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 3 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 4 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 5 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 6 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 7 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 8 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 9 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 10 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 11 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 12 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 13 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 14 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 15 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 16 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 17 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 18 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 19 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 20 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 21 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 22 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 23 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 24 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 25 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 26 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 27 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 28 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 29 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 30 2025</td> <td>775.28</td> </tr> <tr> <td>Mar 31 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 1 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 2 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 3 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 4 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 5 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 6 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 7 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 8 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 9 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 10 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 11 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 12 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 13 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 14 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 15 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 16 2025</td> <td>775.28</td> </tr> <tr> <td>Apr 17 2025</td> <td>775.28</td> </tr> </table>'),
  }),
  map_store_to_wines: new I.Map({
        0: new WineSelection( 0, '/images/clubdvin_image1.jpg', 'Italy', 'Filippo Magnani', 'Discover Italian wines in Oslavia, ', 'Italian wine expert and educator Filippo Magnani will be your guide on this journey to Oslavia, a small hamlet in Collio close to the Slovenian border. '),
        1: new WineSelection( 1, '/images/clubdvin_image2.png', 'France', 'Garth Hodgdon', 'Drink the stars in Champagne, ', 'A late night beer with Olivier Krug, or a private dinner with Peter Liem: nothing is off-limits in our spectacular journey to Champagne with Garth Hodgon.'),
        2: new WineSelection( 2, '/images/clubdvin_image3.png', 'Argentina', 'Valentina Litman', 'Explore the exotic Mendoza, ', 'One of Argentina’s brightest wine talents Valentina Litman and David Garrett will take you on a one-of-a-kind weekend to Mendoza wine country.'),
        3: new WineSelection( 3, '/images/clubdvin_image4.png', 'Germany', 'David Forer', 'Embark down the Mosel, ', 'Explore a breathtaking wine country aboard a boat with the Master of Wine David Forer. Mosel\'s winding river gorge is home to the best Riesling producers')
  }),
  total_stats: Number(0),
  total_trans: Number(0),
};

const modalQRCodeReducer = (state = initialState, action) => {
  switch (action.type) {
    case actionTypes.SET_MODAL_QRCODE_STATUS: {
        // TMD console.log( 'Inside SET_MODAL_QRCODE_STATUS');
        // console.log( {action});
      return {
        ...state,
        isOpen: action.isOpen,
        last_store_id: action.store_id
      };
    }
    case actionTypes.SET_MODAL_QRCODE_LOADING_STATUS: {
        console.log( 'Inside SET_MODAL_QRCODE_LOADING_STATUS');
        console.log( {action});
      return {
        ...state,
        isLoadingOpen: action.isLoadingOpen,
      };
    }
    case actionTypes.SET_MODAL_QRCODE_LOADING_EXEC_STATUS: {
        console.log( 'Inside SET_MODAL_QRCODE_LOADING_EXEC_STATUS');
        console.log( {action});
      return {
        ...state,
        executionStatus: action.executionStatus,
      };
    }
    case actionTypes.SET_MODAL_QRCODE_SCAN: {
        // TMD console.log( 'Inside SET_MODAL_QRCODE_SCAN');
        let login_username = state.login_username;
        let lmap = state.map_store_to_fsop;
        let fsop = lmap.get(action.store_id);
        if ( fsop === undefined)
        {
            console.log( 'SET_MODAL_QRCODE_SCAN: u: ' + login_username + ', store: ' + action.store_id + ', ' + action.stock_price);
            fsop = new CustomerRetailFSOP( 
                    new Customer(login_username, login_username, 'sameer@buybak.xyz', '630-696-7660', 'construction', 'Naperville, IL'), 
                    new Retailer(action.store_id, action.store_id, '171', 'Naperville', 'IL', '60563', map_store_to_images.get(action.store_id)), action.stock_price, 0);
        }
        fsop.stock_price = (fsop.fsop * fsop.stock_price + action.fsop * action.stock_price) / (fsop.fsop + action.fsop);
        fsop.fsop += action.fsop;
        lmap = lmap.set(action.store_id, fsop);
        console.log( 'SET_MODAL_QRCODE_SCAN: Setting Portfolio');
        console.log({fsop});

        let total = Number(0);
        lmap.forEach((item) => {
            total += Number((item.fsop * item.stock_price) / 1000000.00);
        });
        let count = state.alertCount;
        ++count;
        let tdata = state.cdata;
        tdata = tdata.concat(total);
        console.log( 'SCAN: ' + tdata);
      return {
        ...state,
        isLoadingOpen: false,
        last_store_id: action.store_id,
        map_store_to_fsop: lmap,
        total_fsop: total,
        alertCount: count,
        cdata: tdata,
      };
    }
    case actionTypes.SET_MODAL_QRCODE_SELL: {
        // TMD console.log( 'Inside SET_MODAL_QRCODE_SELL');
        let login_username = state.login_username;
        let lmap = state.map_store_to_fsop;
        let fsop = lmap.get(action.store_id);
        if ( fsop === undefined)
        {
            console.log( '!! Error !! SET_MODAL_QRCODE_SELL: u: ' + login_username + ', store: ' + action.store_id + ', ' + action.stock_price);
            return {
                ...state,
                isLoadingOpen: false,
            }
        }
        fsop.fsop -= action.fsop;
        lmap = lmap.set(action.store_id, fsop);
        console.log( 'SET_MODAL_QRCODE_SELL: Setting Portfolio');
        console.log({fsop});

        let total = Number(0);
        lmap.forEach((item) => {
            total += Number((item.fsop * item.stock_price) / 1000000.00);
        });
        let count = state.alertCount;
        ++count;
        let tdata = state.cdata;
        tdata = tdata.concat(total);
        console.log( 'SCAN: ' + tdata);
      return {
        ...state,
        isLoadingOpen: false,
        last_store_id: action.store_id,
        map_store_to_fsop: lmap,
        total_fsop: total,
        alertCount: count,
        cdata: tdata,
      };
    }
    case actionTypes.SET_STOCK_QUOTES: {
        // TMD console.log( 'Inside SET_STOCK_QUOTES');
        // TMD console.log( {action});
        let qmap = state.map_store_to_quotes;
        let qfsop = qmap.get(action.store_id);
        if ( qfsop !== undefined)
        {
            qfsop.stock_price = action.stock_price;
            qfsop.fsop = action.quantity;
            qmap = qmap.set(action.store_id, qfsop);
            // TMD console.log( 'SET_STOCK_QUOTES: set ' + action.stock_price + ' for ' + action.store_id);
        }
        let qtotal = Number(0);
        let lmap = state.map_store_to_fsop;
        lmap.forEach((item) => {
            let qfsop = qmap.get(item.retailer.store_id);
            // TMD console.log({item});
            if ( qfsop !== undefined)
            {
                qtotal += Number((item.fsop * qfsop.stock_price) / 1000000.00);
            }
        });
      return {
        ...state,
        map_store_to_quotes: qmap,
        total_curr: qtotal,
      };
    }
    case actionTypes.SET_BUYBAK_STATISTICS: {
        // TMD console.log( 'Inside SET_BUYBAK_STATISTICS');
        // TMD console.log( {action});
        let qmap = state.map_store_to_stats;
        let qstats = qmap.get(action.store_id);
        if ( qstats === undefined)
        {
            qstats = new BuybakStatistics(action.store_id, action.total_value, action.total_trans);
        }
        qstats.total_value = action.total_value;
        qstats.total_trans = action.total_trans;
        qmap = qmap.set(action.store_id, qstats);
        // TMD console.log( 'SET_BUYBAK_STATISTICS: set ' + action.total_value + ' for ' + action.store_id);

        let qtotal = Number(0);
        qmap.forEach((item) => {
            let qtemp = qmap.get(item.store_id);
            if ( qtemp !== undefined)
            {
                qtotal += qtemp.total_value;
            }
        });
        let ttotal = Number(0);
        qmap.forEach((item) => {
            let qtemp = qmap.get(item.store_id);
            if ( qtemp !== undefined)
            {
                // TMD console.log( item.store_id + ': ' + qtemp.total_trans);
                ttotal += qtemp.total_trans;
            }
        });
        // TMD console.log( 'SET_BUYBAK_STATISTICS: set TotalTx: ' + ttotal);
      return {
        ...state,
        map_store_to_stats: qmap,
        total_stats: qtotal,
        total_trans: ttotal,
      };
    }
    case actionTypes.SET_BUYBAK_RESET_ALERT_COUNT: {
      return {
        ...state,
        alertCount: 0
      };
    }
    case actionTypes.SET_MODAL_MOBILE_LOGIN_STATUS: {
        console.log( 'Inside SET_MODAL_MOBILE_LOGIN_STATUS');
        console.log( {action});
      return {
        ...state,
        is_login_open: action.isOpen,
      };
    }
    case actionTypes.SET_MODAL_MOBILE_LOGIN_NAME: {
        console.log( 'Inside SET_MODAL_MOBILE_LOGIN_NAME');
        console.log( {action});
      return {
        ...state,
        is_login_open: false,
        login_username: action.username
      };
    }
    case actionTypes.SET_BUYBAK_MOBILE_MESSAGE: {
        console.log( 'Inside SET_BUYBAK_MOBILE_MESSAGE');
        console.log( {action});
        let mmap = state.map_store_to_mobile_messages;
        let msg = new MobileChatMessage(action.id, action.user, action.message);
        console.log(msg);
        mmap = mmap.set(action.id, msg);
        console.log( 'SET_BUYBAK_MOBILE_MESSAGE');
        let count = state.alertCount;
        ++count;
        console.log(mmap)
        return {
            ...state,
            map_store_to_mobile_messages: mmap,
            alertCount: count,
        };
    }
    case actionTypes.SET_BUYBAK_PREDICTIONS: {
        console.log( 'Inside SET_BUYBAK_PREDICTIONS');
        console.log( action.values);
        let predictions = state.predictions;
        predictions = action.values;
        return {
            ...state,
            predictions: predictions
        };
    }
    case actionTypes.SET_BUYBAK_FORECASTORS: {
        console.log( 'Inside SET_BUYBAK_FORECASTORS');
        console.log( action.values);
        let forecastors = state.forecastors;
        forecastors = action.values;
        return {
            ...state,
            forecastors: forecastors
        };
    }

    default:
      return state;
  }
};

export default modalQRCodeReducer;

