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
    
    constructor( id: Number, user: String, event_type: String, event_state: String, event_stimuli: String, outline: String, msg: String)
    {
        this.id = id;
        this.user = user;
        this.event_type = event_type;
        this.event_state = event_state;
        this.event_stimuli = event_stimuli;
        this.outline = outline;
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
    'Alsace':  '/images/Alsace.png',
	'Bordeaux':  '/images/Bordauex-3.jpg',
	'Burgundy':  '/images/Burgundy-4.png',
	'Champagne':  '/images/Champagne-3.png',
	'Corsican':  '/images/Corsican.png',
	'Jura':  '/images/Jura-3.png',
	'Savoy':  '/images/Savoy-3.jpg',
	'SouthWestFrench':  '/images/SouthWestFrench.png'
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
  refreshScroll: false,
  map_store_to_quotes: new I.Map({
    'Alsace':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Alsace', 'Alsace', '183', 'Naperville', 'IL', '60563', "/images/Alsace.png"), 26382500, 1199),
    'Bordeaux':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Bordeaux', 'Bordeaux', '115', 'Naperville', 'IL', '60563', "/images/Bordauex-3.jpg"), 971500, 280),
    'Burgundy':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Burgundy', 'Burgundy', '116', 'Naperville', 'IL', '60563', "/images/Burgundy-4.png"), 452500, 1235),
    'Corsican':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Corsican', 'Corsican', '117', 'Naperville', 'IL', '60563', "/images/Corsican-3.png"), 2222600, 543),
    'Champagne':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Champagne', 'Champagne', '118', 'Naperville', 'IL', '60563', "/images/Champagne-3.png"), 7234000, 980),
    'Jura':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Jura', 'Jura', '164', 'Naperville', 'IL', '60563', "/images/Jura-3.png"), 222400, 2211),
    'Savoy':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('Savoy', 'Savoy', '118', 'Naperville', 'IL', '60563', "/images/Savoy-3.jpg"), 7234000, 980),
    'SouthWestFrench':   new CustomerRetailFSOP( 
                    new Customer('', '', '', '', '', ''), 
                    new Retailer('SouthWestFrench', 'SouthWestFrench', '164', 'Naperville', 'IL', '60563', "/images/SouthWestFrench.png"), 222400, 2211),
  }),
  map_store_to_fsop: new I.Map({
    'HD':   new CustomerRetailFSOP( 
                    new Customer('Sameer', '0x1010', 'sameer@buybak.xyz', '630-696-7660', 'construction', 'Naperville, IL'), 
                    new Retailer('HomeDepot', 'HD', '171', 'Naperville', 'IL', '60563', "/images/homedepot.png"), 3631500, 285),
    }),
  list_store_to_mobile_messages: [],
  map_store_to_mobile_messages: new I.Map({
    1745754093973:   new MobileChatMessage( 1745754093973, 'sameer', 'user', '__init', 'InitEvent', '', '<div> <p> Hi ChatGPT</p> </div> '),
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
        console.log( {action});
      return {
        ...state,
        isOpen: action.isOpen,
        last_store_id: action.store_id
      };
    }
    case actionTypes.SET_MODAL_QRCODE_LOADING_STATUS: {
        // TMD console.log( 'Inside SET_MODAL_QRCODE_LOADING_STATUS');
        console.log( {action});
      return {
        ...state,
        isLoadingOpen: action.isLoadingOpen,
      };
    }
    case actionTypes.SET_MODAL_QRCODE_LOADING_EXEC_STATUS: {
        // TMD console.log( 'Inside SET_MODAL_QRCODE_LOADING_EXEC_STATUS');
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
        // TMD console.log( 'Inside SET_MODAL_MOBILE_LOGIN_STATUS');
        console.log( {action});
      return {
        ...state,
        is_login_open: action.isOpen,
      };
    }
    case actionTypes.SET_MODAL_MOBILE_LOGIN_NAME: {
        // TMD console.log( 'Inside SET_MODAL_MOBILE_LOGIN_NAME');
        console.log( {action});
      return {
        ...state,
        is_login_open: false,
        login_username: action.username
      };
    }
    case actionTypes.SET_BUYBAK_MOBILE_MESSAGE: {
        // TMD console.log( 'Inside SET_BUYBAK_MOBILE_MESSAGE');
        console.log( {action});
        let mmap = state.map_store_to_mobile_messages;
        let mlist = state.list_store_to_mobile_messages;
        // Let's JSON parse message
        try {
            // TMD console.log( action.message);
            const jmsg = JSON.parse(action.message);
            // TMD console.log({jmsg});
            const msg = new MobileChatMessage(action.id, action.user, jmsg.event_type, jmsg.event_state, jmsg.event_stimuli, jmsg.event_content.outline, jmsg.event_content.message);
            // TMD console.log(msg);
            mmap = mmap.set(action.id, msg);
            mlist = [...mlist, msg];
        } catch (error) {
            console.log( 'Error: ', error);
        } finally {
            // TMD console.log( 'Finally');
        }
        // TMD console.log( 'SET_BUYBAK_MOBILE_MESSAGE');
        let count = state.alertCount;
        ++count;
        // TMD console.log(mmap)
        return {
            ...state,
            map_store_to_mobile_messages: mmap,
            list_store_to_mobile_messages: mlist,
            alertCount: count,
            refreshScroll: true
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
    case actionTypes.SET_BUYBAK_REFRESH_SCROLL: {
        console.log( 'Inside SET_BUYBAK_REFRESH_SCROLL: ' + action.value);
        return {
            ...state,
            refreshScroll: action.value
        };
    }

    default:
      return state;
  }
};

export default modalQRCodeReducer;

