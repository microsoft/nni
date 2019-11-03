import * as React from 'react';
import { Link } from 'react-router';
import axios from 'axios';
import { MANAGER_IP } from '../static/const';
import MediaQuery from 'react-responsive';
import { Row, Col, Menu, Dropdown, Icon, Select, Button, Form } from 'antd';
import { FormComponentProps } from 'antd/lib/form';
import { OVERVIEWTABS, DETAILTABS, NNILOGO } from './stateless-component/NNItabs';
const { SubMenu } = Menu;
const { Option } = Select;
const FormItem = Form.Item;
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

interface SliderProps extends FormComponentProps {
    changeInterval: (value: number) => void;
}

interface EventPer {
    key: string;
}

class SlideBar extends React.Component<SliderProps, SliderState> {

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
                if (res.status === 200) {
                    this.setState({ version: res.data });
                }
            });
    }

    handleMenuClick = (e: EventPer) => {
        this.setState({ menuVisible: false });
        switch (e.key) {
            // to see & download experiment parameters
            case '1':
                this.setState({ isvisibleExperimentDrawer: true });
                break;
            // to see & download nnimanager log
            case '2':
                this.setState({ activeKey: 'nnimanager', isvisibleLogDrawer: true });
                break;
            // to see & download dispatcher log
            case '3':
                this.setState({ isvisibleLogDrawer: true, activeKey: 'dispatcher' });
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
        this.setState({ menuVisible: flag });
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
            <Menu onClick={this.handleMenuClick} className="menu-list" style={{ width: 216 }}>
                {/* <Menu onClick={this.handleMenuClick} className="menu-list" style={{width: window.innerWidth}}> */}
                <Menu.Item key="feedback">
                    <a href={feedBackLink} target="_blank">Feedback</a>
                </Menu.Item>
                <Menu.Item key="version">Version: {version}</Menu.Item>
                <SubMenu
                    key="download"
                    onChange={this.handleVisibleChange}
                    title={
                        <span>
                            <span>View</span>
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

    mobileTabs = () => {
        return (
            // <Menu className="menuModal" style={{width: 880, position: 'fixed', left: 0, top: 56}}>
            <Menu className="menuModal" style={{ padding: '0 10px' }}>
                <Menu.Item key="overview"><Link to={'/oview'}>Overview</Link></Menu.Item>
                <Menu.Item key="detail"><Link to={'/detail'}>Trials detail</Link></Menu.Item>
            </Menu>
        );
    }

    refreshInterval = () => {
        const {
            form: { getFieldDecorator },
            // form: { getFieldDecorator, getFieldValue },
        } = this.props;
        return (
            <Form>
                <FormItem style={{ marginBottom: 0 }}>
                    {getFieldDecorator('interval', {
                        initialValue: 'Refresh every 10s',
                    })(
                        <Select onSelect={this.getInterval}>
                            <Option value="close">Disable Auto Refresh</Option>
                            <Option value="10">Refresh every 10s</Option>
                            <Option value="20">Refresh every 20s</Option>
                            <Option value="30">Refresh every 30s</Option>
                            <Option value="60">Refresh every 1min</Option>
                        </Select>,
                    )}
                </FormItem>
            </Form>
        );
    }

    select = () => {
        const { isdisabledFresh } = this.state;

        return (
            <div className="interval">
                {this.refreshInterval()}
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
        this.setState({ isdisabledFresh: true }, () => {
            setTimeout(() => { this.setState({ isdisabledFresh: false }); }, 1000);
        });
    }

    desktopHTML = () => {
        const { version, menuVisible } = this.state;
        const feed = `https://github.com/Microsoft/nni/issues/new?labels=${version}`;
        return (
            <Row className="nav">
                <Col span={8}>
                    <span className="desktop-logo">{NNILOGO}</span>
                    <span className="left-right-margin">{OVERVIEWTABS}</span>
                    <span>{DETAILTABS}</span>
                </Col>
                <Col span={16} className="desktop-right">
                    <span>
                        <Button
                            className="fresh"
                            type="ghost"
                        >
                            <a target="_blank" href="https://nni.readthedocs.io/en/latest/Tutorial/WebUI.html">
                                <img
                                    src={require('../static/img/icon/ques.png')}
                                    alt="question"
                                    className="question"
                                />
                                <span>Help</span>
                            </a>
                        </Button>
                    </span>
                    <span>{this.select()}</span>
                    <span>
                        <Dropdown
                            className="dropdown"
                            overlay={this.menu()}
                            onVisibleChange={this.handleVisibleChange}
                            visible={menuVisible}
                            trigger={['click']}
                        >
                            <a className="ant-dropdown-link" href="#">
                                <Icon type="download" className="down-icon" />
                                <span>View</span>
                                {
                                    menuVisible
                                        ?
                                        <Icon type="up" className="margin-icon" />
                                        :
                                        <Icon type="down" className="margin-icon" />
                                }
                            </a>
                        </Dropdown>
                    </span>
                    <span className="feedback">
                        <a href={feed} target="_blank">
                            <img
                                src={require('../static/img/icon/issue.png')}
                                alt="NNI github issue"
                            />
                            Feedback
                        </a>
                    </span>
                    <span className="version">Version: {version}</span>
                </Col>
            </Row>
        );
    }

    tabeltHTML = () => {
        return (
            <Row className="nav">
                <Col className="tabelt-left" span={14}>
                    <span>
                        <Dropdown overlay={this.navigationBar()} trigger={['click']}>
                            <Icon type="unordered-list" className="more" />
                        </Dropdown>
                    </span>
                    <span className="left-right-margin tabelt-img">{NNILOGO}</span>
                    <span>{OVERVIEWTABS}</span>
                    <span className="left-margin">{DETAILTABS}</span>
                </Col>
                <Col className="tabelt-right" span={10}>
                    {this.select()}
                </Col>
            </Row>
        );
    }

    mobileHTML = () => {
        const { isdisabledFresh } = this.state;
        return (
            <Row className="nav">
                <Col className="left" span={8}>
                    <span>
                        <Dropdown className="more-mobile" overlay={this.navigationBar()} trigger={['click']}>
                            <Icon type="unordered-list" className="more" />
                        </Dropdown>
                    </span>
                    <span>
                        <Dropdown overlay={this.mobileTabs()} trigger={['click']}>
                            <a className="ant-dropdown-link" href="#">
                                <span>NNI <Icon type="down" /></span>
                            </a>
                        </Dropdown>
                    </span>
                </Col>
                <Col className="center" span={8}>
                    <img
                        src={require('../static/img/logo2.png')}
                        alt="NNI logo"
                    />
                </Col>
                {/* the class interval have other style ! */}
                <Col className="right interval" span={8}>
                    <Button
                        className="fresh"
                        onClick={this.fresh}
                        type="ghost"
                        disabled={isdisabledFresh}
                    >
                        <Icon type="sync" /><span>Refresh</span>
                    </Button>
                </Col>
            </Row>
        );
    }
    // close log drawer (nnimanager.dispatcher)
    closeLogDrawer = () => {
        this.setState({ isvisibleLogDrawer: false, activeKey: '' });
    }

    // close download experiment parameters drawer
    closeExpDrawer = () => {
        this.setState({ isvisibleExperimentDrawer: false });
    }

    componentDidMount() {
        this.getNNIversion();
    }

    render() {
        const mobile = (<MediaQuery maxWidth={884}>{this.mobileHTML()}</MediaQuery>);
        const tablet = (<MediaQuery minWidth={885} maxWidth={1281}>{this.tabeltHTML()}</MediaQuery>);
        const desktop = (<MediaQuery minWidth={1282}>{this.desktopHTML()}</MediaQuery>);
        const { isvisibleLogDrawer, activeKey, isvisibleExperimentDrawer } = this.state;
        return (
            <div>
                {mobile}
                {tablet}
                {desktop}
                {/* the drawer for dispatcher & nnimanager log message */}
                {isvisibleLogDrawer ? (
                    <LogDrawer
                        closeDrawer={this.closeLogDrawer}
                        activeTab={activeKey}
                    />
                ) : null}
                <ExperimentDrawer
                    isVisble={isvisibleExperimentDrawer}
                    closeExpDrawer={this.closeExpDrawer}
                />
            </div>
        );
    }
}

export default Form.create<FormComponentProps>()(SlideBar);
