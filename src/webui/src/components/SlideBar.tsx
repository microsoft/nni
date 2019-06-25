import * as React from 'react';
import { Link } from 'react-router';
import axios from 'axios';
import { MANAGER_IP } from '../static/const';
import MediaQuery from 'react-responsive';
import { DOWNLOAD_IP } from '../static/const';
import { Row, Col, Menu, Dropdown, Icon, Select } from 'antd';
const { SubMenu } = Menu;
const { Option } = Select;
import '../static/style/slideBar.scss';
import '../static/style/button.scss';

interface SliderState {
    version: string;
    menuVisible: boolean;
    navBarVisible: boolean;
}

interface SliderProps {
    changeInterval: (value: number) => void;
    changeFresh: (value: string) => void;
}

interface EventPer {
    key: string;
}

class SlideBar extends React.Component<SliderProps, SliderState> {

    public _isMounted = false;
    public divMenu: HTMLDivElement | null;
    public selectHTML: Select | null;

    constructor(props: SliderProps) {
        super(props);
        this.state = {
            version: '',
            menuVisible: false,
            navBarVisible: false,
        };
    }

    downExperimentContent = () => {
        axios
            .all([
                axios.get(`${MANAGER_IP}/experiment`),
                axios.get(`${MANAGER_IP}/trial-jobs`),
                axios.get(`${MANAGER_IP}/metric-data`)
            ])
            .then(axios.spread((res, res1, res2) => {
                if (res.status === 200 && res1.status === 200 && res2.status === 200) {
                    if (res.data.params.searchSpace) {
                        res.data.params.searchSpace = JSON.parse(res.data.params.searchSpace);
                    }
                    const isEdge = navigator.userAgent.indexOf('Edge') !== -1 ? true : false;
                    let trialMessagesArr = res1.data;
                    const interResultList = res2.data;
                    Object.keys(trialMessagesArr).map(item => {
                        // transform hyperparameters as object to show elegantly
                        trialMessagesArr[item].hyperParameters = JSON.parse(trialMessagesArr[item].hyperParameters);
                        const trialId = trialMessagesArr[item].id;
                        // add intermediate result message
                        trialMessagesArr[item].intermediate = [];
                        Object.keys(interResultList).map(key => {
                            const interId = interResultList[key].trialJobId;
                            if (trialId === interId) {
                                trialMessagesArr[item].intermediate.push(interResultList[key]);
                            }
                        });
                    });
                    const result = {
                        experimentParameters: res.data,
                        trialMessage: trialMessagesArr
                    };
                    const aTag = document.createElement('a');
                    const file = new Blob([JSON.stringify(result, null, 4)], { type: 'application/json' });
                    aTag.download = 'experiment.json';
                    aTag.href = URL.createObjectURL(file);
                    aTag.click();
                    if (!isEdge) {
                        URL.revokeObjectURL(aTag.href);
                    }
                    if (navigator.userAgent.indexOf('Firefox') > -1) {
                        const downTag = document.createElement('a');
                        downTag.addEventListener('click', function () {
                            downTag.download = 'experiment.json';
                            downTag.href = URL.createObjectURL(file);
                        });
                        let eventMouse = document.createEvent('MouseEvents');
                        eventMouse.initEvent('click', false, false);
                        downTag.dispatchEvent(eventMouse);
                    }
                }
            }));
    }

    downnnimanagerLog = () => {
        axios(`${DOWNLOAD_IP}/nnimanager.log`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const nniLogfile = res.data;
                    const aTag = document.createElement('a');
                    const isEdge = navigator.userAgent.indexOf('Edge') !== -1 ? true : false;
                    const file = new Blob([nniLogfile], { type: 'application/json' });
                    aTag.download = 'nnimanagerLog.log';
                    aTag.href = URL.createObjectURL(file);
                    aTag.click();
                    if (!isEdge) {
                        URL.revokeObjectURL(aTag.href);
                    }
                    if (navigator.userAgent.indexOf('Firefox') > -1) {
                        const downTag = document.createElement('a');
                        downTag.addEventListener('click', function () {
                            downTag.download = 'nnimanagerLog.log';
                            downTag.href = URL.createObjectURL(file);
                        });
                        let eventMouse = document.createEvent('MouseEvents');
                        eventMouse.initEvent('click', false, false);
                        downTag.dispatchEvent(eventMouse);
                    }
                }
            });
    }

    downDispatcherlog = () => {
        axios(`${DOWNLOAD_IP}/dispatcher.log`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const dispatchLogfile = res.data;
                    const aTag = document.createElement('a');
                    const isEdge = navigator.userAgent.indexOf('Edge') !== -1 ? true : false;
                    const file = new Blob([dispatchLogfile], { type: 'application/json' });
                    aTag.download = 'dispatcherLog.log';
                    aTag.href = URL.createObjectURL(file);
                    aTag.click();
                    if (!isEdge) {
                        URL.revokeObjectURL(aTag.href);
                    }
                    if (navigator.userAgent.indexOf('Firefox') > -1) {
                        const downTag = document.createElement('a');
                        downTag.addEventListener('click', function () {
                            downTag.download = 'dispatcherLog.log';
                            downTag.href = URL.createObjectURL(file);
                        });
                        let eventMouse = document.createEvent('MouseEvents');
                        eventMouse.initEvent('click', false, false);
                        downTag.dispatchEvent(eventMouse);
                    }
                }
            });
    }

    getNNIversion = () => {
        axios(`${MANAGER_IP}/version`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200 && this._isMounted) {
                    this.setState({ version: res.data });
                }
            });
    }

    handleMenuClick = (e: EventPer) => {
        if (this._isMounted) { this.setState({ menuVisible: false }); }
        switch (e.key) {
            // download experiment related content
            case '1':
                this.downExperimentContent();
                break;
            // download nnimanager log file
            case '2':
                this.downnnimanagerLog();
                break;
            // download dispatcher log file
            case '3':
                this.downDispatcherlog();
                break;
            case 'close':
            case '10':
            case '20':
            case '30':
            case '60':
                this.getInterval(e.key);
                break;
            default:
        }
    }

    handleVisibleChange = (flag: boolean) => {
        if (this._isMounted === true) {
            this.setState({ menuVisible: flag });
        }
    }

    getInterval = (value: string) => {

        if (value === 'close') {
            this.props.changeInterval(0);
        } else {
            this.props.changeInterval(parseInt(value, 10));
        }
    }

    menu = () => {
        return (
            <Menu onClick={this.handleMenuClick}>
                <Menu.Item key="1">Experiment Parameters</Menu.Item>
                <Menu.Item key="2">NNImanager Logfile</Menu.Item>
                <Menu.Item key="3">Dispatcher Logfile</Menu.Item>
            </Menu>
        );
    }

    // nav bar
    navigationBar = () => {
        const { version } = this.state;
        const feedBackLink = `https://github.com/Microsoft/nni/issues/new?labels=${version}`;
        return (
            <Menu onClick={this.handleMenuClick} className="menuModal">
                <Menu.Item key="overview"><Link to={'/oview'}>Overview</Link></Menu.Item>
                <Menu.Item key="detail"><Link to={'/detail'}>Trials detail</Link></Menu.Item>
                <Menu.Item key="fresh">
                    <span className="fresh" onClick={this.fresh}><span>Fresh</span></span>
                </Menu.Item>
                <Menu.Item key="feedback">
                    <a href={feedBackLink} target="_blank">Feedback</a>
                </Menu.Item>
                <Menu.Item key="version">Version: {version}</Menu.Item>
                <SubMenu
                    key="download"
                    onChange={this.handleVisibleChange}
                    title={
                        <span>
                            <span>Download</span>
                        </span>
                    }
                >
                    <Menu.Item key="1">Experiment Parameters</Menu.Item>
                    <Menu.Item key="2">NNImanager Logfile</Menu.Item>
                    <Menu.Item key="3">Dispatcher Logfile</Menu.Item>
                </SubMenu>
            </Menu>
        );
    }

    select = () => {
        return (
            <Select
                onSelect={this.getInterval}
                defaultValue="Refresh every 10s"
                className="interval"
            >
                <Option value="close">Disable Auto Refresh</Option>
                <Option value="10">Refresh every 10s</Option>
                <Option value="20">Refresh every 20s</Option>
                <Option value="30">Refresh every 30s</Option>
                <Option value="60">Refresh every 1min</Option>
            </Select>
        );
    }

    fresh = (event: React.SyntheticEvent<EventTarget>) => {
        event.preventDefault();
        const whichPage = window.location.pathname;
        this.props.changeFresh(whichPage);
    }

    componentDidMount() {
        this._isMounted = true;
        this.getNNIversion();
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {
        const { version, menuVisible } = this.state;
        const feed = `https://github.com/Microsoft/nni/issues/new?labels=${version}`;
        return (
            <Row>
                <MediaQuery query="(min-width: 1299px)">
                    <Row className="nav">
                        <ul className="link">
                            <li className="logo">
                                <Link to={'/oview'}>
                                    <img
                                        src={require('../static/img/logo2.png')}
                                        style={{ width: 88 }}
                                        alt="NNI logo"
                                    />
                                </Link>
                            </li>
                            <li className="tab firstTab">
                                <Link to={'/oview'} activeClassName="high">
                                    Overview
                                    </Link>
                            </li>
                            <li className="tab">
                                <Link to={'/detail'} activeClassName="high">
                                    Trials detail
                                    </Link>
                            </li>
                            <li className="feedback">
                                <span className="fresh" onClick={this.fresh}>
                                    <Icon type="sync" /><span>Fresh</span>
                                </span>
                                <Dropdown
                                    className="dropdown"
                                    overlay={this.menu()}
                                    onVisibleChange={this.handleVisibleChange}
                                    visible={menuVisible}
                                    trigger={['click']}
                                >
                                    <a className="ant-dropdown-link" href="#">
                                        Download <Icon type="down" />
                                    </a>
                                </Dropdown>
                                <a href={feed} target="_blank">
                                    <img
                                        src={require('../static/img/icon/issue.png')}
                                        alt="NNI github issue"
                                    />
                                    Feedback
                            </a>
                                <span className="version">Version: {version}</span>
                            </li>
                        </ul>
                    </Row>
                </MediaQuery>
                <MediaQuery query="(max-width: 1299px)">
                    <Row className="little">
                        <Col span={6} className="menu">
                            <Dropdown overlay={this.navigationBar()} trigger={['click']}>
                                <Icon type="unordered-list" className="more" />
                            </Dropdown>
                        </Col>
                        <Col span={10} className="logo">
                            <Link to={'/oview'}>
                                <img
                                    src={require('../static/img/logo2.png')}
                                    style={{ width: 88 }}
                                    alt="NNI logo"
                                />
                            </Link>
                        </Col>
                    </Row>
                </MediaQuery>
                {this.select()}
            </Row>
        );
    }
}

export default SlideBar;