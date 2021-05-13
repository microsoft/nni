import { mergeStyleSets } from '@fluentui/react';

const classNames = mergeStyleSets({
    menu: {
        textAlign: 'center',
        maxWidth: 600,
        selectors: {
            '.ms-ContextualMenu-item': {
                height: 'auto'
            }
        }
    },
    item: {
        display: 'inline-block',
        width: 40,
        height: 40,
        lineHeight: 40,
        textAlign: 'center',
        verticalAlign: 'middle',
        marginBottom: 8,
        cursor: 'pointer',
        selectors: {
            '&:hover': {
                backgroundColor: '#eaeaea'
            }
        }
    },
    categoriesList: {
        margin: 0,
        padding: 0,
        listStyleType: 'none'
    },
    button: {
        width: '40%',
        margin: '2%'
    }
});

export { classNames };
