import * as React from 'react';
import { Link } from 'react-router';
import axios from 'axios';
import { MANAGER_IP } from '../static/const';
import MediaQuery from 'react-responsive';
import { Row, Col, Menu, Dropdown, Icon, Select, Button } from 'antd';
const { SubMenu } = Menu;
const { Option } = Select;
import LogDrawer from './Modal/LogDrawer';
import ExperimentDrawer from './Modal/ExperimentDrawer';
import '../static/style/slideBar.scss';
import '../static/style/button.scss';

interface SliderState {
    version: string;
    menuVisible: boolean;
    navBarVisible: boolean;
    isdisabledFresh: boolean;
    isvisibleLogDrawer: boolean;
    isvisibleExperimentDrawer: boolean;
    activeKey: string;
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
            isdisabledFresh: false,
            isvisibleLogDrawer: false, // download button (nnimanager·dispatcher) click -> drawer
            isvisibleExperimentDrawer: false,
            activeKey: 'dispatcher'
        };
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
            // to see & download experiment parameters
            case '1':
                if (this._isMounted === true) {
                    this.setState(() => ({ isvisibleExperimentDrawer: true }));
                }
                break;
            // to see & download nnimanager log
            case '2':
                if (this._isMounted === true) {
                    this.setState(() => ({ activeKey: 'nnimanager', isvisibleLogDrawer: true }));
                }
                break;
            // to see & download dispatcher log
            case '3':
                if (this._isMounted === true) {
                    this.setState(() => ({ isvisibleLogDrawer: true, activeKey: 'dispatcher' }));
                }
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
        const { isdisabledFresh } = this.state;
        return (
            <div className="interval">
                <Select
                    onSelect={this.getInterval}
                    defaultValue="Refresh every 10s"
                >
                    <Option value="close">Disable Auto Refresh</Option>
                    <Option value="10">Refresh every 10s</Option>
                    <Option value="20">Refresh every 20s</Option>
                    <Option value="30">Refresh every 30s</Option>
                    <Option value="60">Refresh every 1min</Option>
                </Select>
                <Button
                    className="fresh"
                    onClick={this.fresh}
                    type="ghost"
                    disabled={isdisabledFresh}
                >
                    <Icon type="sync" /><span>Refresh</span>
                </Button>
            </div>
        );
    }

    fresh = (event: React.SyntheticEvent<EventTarget>) => {
        event.preventDefault();
        event.stopPropagation();
        if (this._isMounted) {
            this.setState({ isdisabledFresh: true }, () => {
                const whichPage = window.location.pathname;
                this.props.changeFresh(whichPage);
                setTimeout(() => { this.setState(() => ({ isdisabledFresh: false })); }, 1000);
            });
        }
    }

    // close log drawer (nnimanager.dispatcher)
    closeLogDrawer = () => {
        if (this._isMounted === true) {
            this.setState(() => ({ isvisibleLogDrawer: false, activeKey: '' }));
        }
    }

    // close download experiment parameters drawer
    closeExpDrawer = () => {
        if (this._isMounted === true) {
            this.setState(() => ({ isvisibleExperimentDrawer: false }));
        }
    }

    componentDidMount() {
        this._isMounted = true;
        this.getNNIversion();
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {
        const { version, menuVisible, isvisibleLogDrawer, activeKey, isvisibleExperimentDrawer } = this.state;
        const feed = `https://github.com/Microsoft/nni/issues/new?labels=${version}`;
        return (
            <Row>
                <Row>
                    <Col span={18}>
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
                    </Col>
                    <Col span={18}>
                        <MediaQuery query="(max-width: 1299px)">
                            <Row className="little">
                                <Col span={1} className="menu">
                                    <Dropdown overlay={this.navigationBar()} trigger={['click']}>
                                        <Icon type="unordered-list" className="more" />
                                    </Dropdown>
                                </Col>
                                <Col span={14} className="logo">
                                    <Link to={'/oview'}>
                                        <img
                                            src={require('../static/img/logo2.png')}
                                            style={{ width: 80 }}
                                            alt="NNI logo"
                                        />
                                    </Link>
                                </Col>
                            </Row>
                        </MediaQuery>
                    </Col>
                    <Col span={3}> {this.select()} </Col>
                </Row>
                {/* the drawer for dispatcher & nnimanager log message */}
                <LogDrawer
                    isVisble={isvisibleLogDrawer}
                    closeDrawer={this.closeLogDrawer}
                    activeTab={activeKey}
                />
                <ExperimentDrawer
                    isVisble={isvisibleExperimentDrawer}
                    closeExpDrawer={this.closeExpDrawer}
                />
            </Row>
        );
    }
}

export default SlideBar;