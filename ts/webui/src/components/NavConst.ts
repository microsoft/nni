import { IStackTokens, IStackStyles } from '@fluentui/react';

const stackTokens: IStackTokens = {
    childrenGap: 15
};
const stackStyle: IStackStyles = {
    root: {
        minWidth: 400,
        height: 56,
        display: 'flex',
        verticalAlign: 'center'
    }
};

export { stackTokens, stackStyle };
